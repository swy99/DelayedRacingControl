"""
변형 아키텍처 학습 — delay × arch × init 그리드의 한 셀.

arch ∈ {gru, actstack, obsstack, both}   (naive는 855 재사용, 학습 안 함)
init ∈ {scratch, pretrain}
  scratch  : 무작위 초기화부터 delay=k 학습
  pretrain : 같은 arch의 delay=0 모델(scratch, T₀)에서 warm-start 후 delay=k 학습
             — 스택 크기 고정 20이라 아키텍처가 delay에 불변 → 완전 가중치 로드

수렴 정지: EvalCallback + StopTrainingOnNoModelImprovement(8, min_evals=10), 상한 5M.
저장: models/variants/<arch>/<init>/d<delay>/best_model.zip (EvalCallback이 best 저장).

실행 (프로젝트 루트에서):
    python src/train_variant.py --arch gru  --delay 0  --init scratch    # T₀ 소스
    python src/train_variant.py --arch both --delay 10 --init scratch
    python src/train_variant.py --arch both --delay 10 --init pretrain   # d0에서 warm-start
"""
import argparse

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    EvalCallback, StopTrainingOnNoModelImprovement, CallbackList,
)
from stable_baselines3.common.utils import get_schedule_fn

import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True,
                        choices=[a for a in common.ARCH_SPEC if a != "naive"])
    parser.add_argument("--delay", type=int, required=True)
    parser.add_argument("--init", choices=["scratch", "pretrain"], default="scratch")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.init == "pretrain" and args.delay == 0:
        raise SystemExit("delay=0은 pretrain 소스(T₀)이므로 scratch로 학습한다.")

    spec = common.ARCH_SPEC[args.arch]
    Algo = RecurrentPPO if spec["recurrent"] else PPO
    device = common.get_device(args.device)
    cell = f"{args.arch}_d{args.delay}_{args.init}"
    print(f"=== {cell} | {Algo.__name__} / {spec['policy']} | "
          f"n_obs={spec['n_obs']} n_act={spec['n_act']} n_stack={spec['n_stack']} ===")

    save_dir = common.variant_dir(common.MODELS_DIR, args.arch, args.init, args.delay)
    log_dir  = common.variant_dir(common.RUNS_DIR,   args.arch, args.init, args.delay)
    tb = str(common.RUNS_DIR / "tb_logs")

    # ── 환경 ──────────────────────────────────────────────────────────
    train_env = common.make_arch_env(args.n_envs, args.arch, args.delay,
                                     monitor_dir=log_dir)
    eval_env  = common.make_arch_env(4, args.arch, args.delay)

    # ── 콜백 (수렴 정지) ──────────────────────────────────────────────
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=8, min_evals=10, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(log_dir),
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        callback_after_eval=stop_cb,
        verbose=1,
    )

    # ── 모델 ──────────────────────────────────────────────────────────
    if args.init == "pretrain":
        src = common.variant_dir(common.MODELS_DIR, args.arch, "scratch", 0) / "best_model"
        print(f"pretrain ← {src}")
        model = Algo.load(str(src), env=train_env, device=device, tensorboard_log=tb)
        model.learning_rate = common.linear_schedule(args.lr)
        model.lr_schedule = get_schedule_fn(model.learning_rate)
        model.target_kl = 0.02
    else:
        model = Algo(
            spec["policy"], train_env,
            n_steps=512, batch_size=256,
            learning_rate=common.linear_schedule(args.lr),
            target_kl=0.02,
            tensorboard_log=tb, device=device, verbose=1,
        )

    # ── 학습 ──────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([eval_cb]),
        reset_num_timesteps=True,
        tb_log_name=cell,
        progress_bar=True,
    )
    model.save(str(save_dir / "final_model"))
    print(f"저장 완료: {save_dir}  (best_model.zip = peak eval, final_model.zip = 종료시점)")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
