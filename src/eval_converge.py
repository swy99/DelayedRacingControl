"""
딜레이 실험 (수렴 기반 평가) — naive(855) baseline 전용.

고정 에피소드 수가 아니라 SEM(표준오차) 기반 정지 규칙으로 평균이 수렴할
때까지 에피소드를 추가한다. delay마다 분산이 다르므로 필요한 에피소드 수도
자동으로 달라진다.

수렴 루프·요약 통계는 common.collect_until_converged / common.summarize 공유.
변형 아키텍처 셀 평가는 eval_variant.py 참고.

실행 (프로젝트 루트에서):
    python src/eval_converge.py --delays 0 3 6 10 15 20 --sem-target 5.0
"""
import argparse
import json
import numpy as np
from stable_baselines3 import PPO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common


def evaluate(model_path, delays, n_envs, min_ep, max_ep, sem_target, patience):
    model = PPO.load(model_path)
    print(f"\n수렴 기준: SEM < {sem_target} (min={min_ep}, max={max_ep}, "
          f"patience={patience}, n_envs={n_envs})\n")

    results = {}
    for delay in delays:
        print(f"── delay={delay} 수집 시작 ──")
        env = common.make_carracing_vec(n_envs, delay=delay, seed=0, transpose=True)
        rewards, history, converged_at = common.collect_until_converged(
            model, env, n_envs, recurrent=False,
            min_ep=min_ep, max_ep=max_ep, sem_target=sem_target,
            patience=patience, label=f"delay={delay:>2}")
        summary = common.summarize(rewards, converged_at, delay=int(delay))
        summary["history"] = history
        results[delay] = summary
        tag = (f"수렴 (n={converged_at})" if converged_at
               else f"미수렴 (max_ep={max_ep} 도달)")
        print(f"  → {tag}: mean={summary['mean']:.1f} "
              f"± {summary['ci95_halfwidth']:.1f} (95% CI)\n")

    print("── 최종 요약 ─────────────────────────────────────────────")
    print(f"{'delay':>6} | {'n':>4} | {'mean':>8} | {'95%CI±':>7} | "
          f"{'fail%':>6} | 수렴")
    print("-" * 56)
    for d, s in results.items():
        print(f"{d:>6} | {s['n']:>4} | {s['mean']:>8.1f} | "
              f"{s['ci95_halfwidth']:>7.1f} | {s['fail_rate']*100:>5.0f}% | "
              f"{'O' if s['converged'] else 'X'}")

    return results


def plot(results, save_path="delay_converged.png"):
    delays = list(results.keys())
    means = [results[d]["mean"] for d in delays]
    ci = [results[d]["ci95_halfwidth"] for d in delays]
    base = means[0] if means[0] != 0 else 1e-9
    deg = [(m - base) / abs(base) * 100 for m in means]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Action Delay vs Performance — Converged Evaluation "
                 "(CarRacing-v3, PPO)", fontsize=14, fontweight="bold")

    # ── 패널 1: 평균 ± 95% CI ─────────────────────────────────────
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(delays)))
    ax1.bar(delays, means, yerr=ci, capsize=6, color=colors,
            edgecolor="k", linewidth=0.6, width=1.5)
    for d in delays:
        rews = results[d]["rewards"]
        ax1.scatter([d] * len(rews), rews, color="k", s=6, alpha=0.25, zorder=5)
    for d, m, c in zip(delays, means, ci):
        n = results[d]["n"]
        ax1.text(d, m + c + 8, f"{m:.0f}\n(n={n})", ha="center", va="bottom",
                 fontsize=8, fontweight="bold")
    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    ax1.set_xlabel("Action Delay (steps)")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Mean ± 95% CI  (dots = individual eps)")
    ax1.set_xticks(delays)

    # ── 패널 2: 상대 성능 저하 ────────────────────────────────────
    ax2.plot(delays, deg, "o-", color="steelblue", lw=2, ms=7)
    ax2.fill_between(delays, deg, alpha=0.12, color="steelblue")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    for d, g in zip(delays, deg):
        ax2.annotate(f"{g:+.0f}%", (d, g), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8,
                     color="steelblue", fontweight="bold")
    ax2.set_xlabel("Action Delay (steps)")
    ax2.set_ylabel("Change vs delay=0 (%)")
    ax2.set_title("Relative Degradation")
    ax2.set_xticks(delays)
    ax2.grid(axis="y", alpha=0.3)

    # ── 패널 3: 수렴 곡선 (SEM vs n) ──────────────────────────────
    for d, color in zip(delays, colors):
        hist = results[d]["history"]
        if hist:
            ns = [h[0] for h in hist]
            sems = [h[1] for h in hist]
            ax3.plot(ns, sems, lw=1.5, color=color, label=f"delay={d}")
    ax3.set_xlabel("Episodes collected (n)")
    ax3.set_ylabel("SEM = std / √n")
    ax3.set_title("Convergence (lower = more converged)")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n그래프 저장: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default=str(common.DEFAULT_MODEL))
    parser.add_argument("--delays",     nargs="+", type=int,
                        default=[0, 3, 6, 10, 15, 20])
    parser.add_argument("--n-envs",     type=int,   default=8)
    parser.add_argument("--min-ep",     type=int,   default=30)
    parser.add_argument("--max-ep",     type=int,   default=400)
    parser.add_argument("--sem-target", type=float, default=5.0)
    parser.add_argument("--patience",   type=int,   default=16)
    parser.add_argument("--out",        type=str,
                        default=str(common.RESULTS_DIR / "delay_converged.png"))
    parser.add_argument("--json-out",   type=str,
                        default=str(common.RESULTS_DIR / "delay_converged.json"))
    args = parser.parse_args()

    results = evaluate(args.model, args.delays, args.n_envs,
                       args.min_ep, args.max_ep, args.sem_target, args.patience)
    plot(results, save_path=args.out)

    # 원시 데이터 저장 (history는 용량 큰 plot용이라 제외)
    dump = {str(d): {k: v for k, v in s.items() if k != "history"}
            for d, s in results.items()}
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    print(f"원시 데이터 저장: {args.json_out}")
