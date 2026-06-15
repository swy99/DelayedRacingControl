"""
변형 아키텍처 셀 평가 — best_model을 그 셀의 delay에서 SEM 수렴 평가.

common.collect_until_converged(recurrent 지원) + common.summarize 재사용.
셀 결과를 results/variant_results.json에 누적 저장하고, naive(855) 기준선과 함께
delay축 비교 그래프(results/variant_compare.png)를 그린다.

실행 (프로젝트 루트에서):
    python src/eval_variant.py --arch both --delay 10 --init scratch
    python src/eval_variant.py --plot-only           # 저장된 json으로 그래프만
"""
import argparse
import json

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common

RESULTS_JSON = common.RESULTS_DIR / "variant_results.json"


def eval_cell(arch, init, delay, n_envs, min_ep, max_ep, sem_target, patience):
    spec = common.ARCH_SPEC[arch]
    Algo = RecurrentPPO if spec["recurrent"] else PPO
    path = common.variant_dir(common.MODELS_DIR, arch, init, delay) / "best_model"
    model = Algo.load(str(path))
    cell = f"{arch}_d{delay}_{init}"
    print(f"── {cell} 평가 (SEM<{sem_target}) ──")

    env = common.make_arch_env(n_envs, arch, delay, seed=0, transpose=True)
    rewards, history, converged_at = common.collect_until_converged(
        model, env, n_envs, recurrent=spec["recurrent"],
        min_ep=min_ep, max_ep=max_ep, sem_target=sem_target,
        patience=patience, label=cell)
    summary = common.summarize(
        rewards, converged_at, cell=cell, arch=arch, init=init, delay=int(delay))
    tag = f"수렴(n={converged_at})" if converged_at else f"미수렴(max={max_ep})"
    print(f"  → {tag}: {summary['mean']:.1f} ± {summary['ci95_halfwidth']:.1f} "
          f"(95% CI) | fail {summary['fail_rate']*100:.0f}%\n")
    return summary


def load_results():
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_result(summary):
    data = load_results()
    data[summary["cell"]] = {k: v for k, v in summary.items() if k != "rewards"}
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {RESULTS_JSON}  ({summary['cell']})")


def naive_baseline():
    """results/delay_converged.json(855 delay축 평가)에서 delay→mean 맵을 읽는다."""
    path = common.RESULTS_DIR / "delay_converged.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {int(k): v["mean"] for k, v in d.items()}


def plot(save_path):
    data = load_results()
    if not data:
        print("결과 없음 — 평가를 먼저 실행하라.")
        return
    archs = sorted({v["arch"] for v in data.values()})
    inits = sorted({v["init"] for v in data.values()})
    base = naive_baseline()

    fig, ax = plt.subplots(figsize=(10, 6))
    if base:
        bd = sorted(base)
        ax.plot(bd, [base[d] for d in bd], "k--o", lw=2,
                label="naive (855, reused)", zorder=10)
    markers = {"scratch": "o", "pretrain": "s"}
    for arch in archs:
        for init in inits:
            pts = sorted((v for v in data.values()
                          if v["arch"] == arch and v["init"] == init),
                         key=lambda v: v["delay"])
            if not pts:
                continue
            xs = [p["delay"] for p in pts]
            ys = [p["mean"] for p in pts]
            es = [p["ci95_halfwidth"] for p in pts]
            ax.errorbar(xs, ys, yerr=es, marker=markers.get(init, "o"),
                        capsize=3, lw=1.5, label=f"{arch}/{init}")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Action Delay (steps)")
    ax.set_ylabel("Converged Episode Reward (± 95% CI)")
    ax.set_title("Delay Mitigation by Architecture (CarRacing-v3)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"그래프 저장: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=[a for a in common.ARCH_SPEC if a != "naive"])
    p.add_argument("--delay", type=int)
    p.add_argument("--init", choices=["scratch", "pretrain"], default="scratch")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--min-ep", type=int, default=30)
    p.add_argument("--max-ep", type=int, default=400)
    p.add_argument("--sem-target", type=float, default=5.0)
    p.add_argument("--patience", type=int, default=16)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--out", type=str,
                   default=str(common.RESULTS_DIR / "variant_compare.png"))
    args = p.parse_args()

    if not args.plot_only:
        if args.arch is None or args.delay is None:
            p.error("--arch 와 --delay 가 필요하다 (또는 --plot-only).")
        summary = eval_cell(args.arch, args.init, args.delay, args.n_envs,
                            args.min_ep, args.max_ep, args.sem_target, args.patience)
        save_result(summary)
    plot(args.out)


if __name__ == "__main__":
    main()
