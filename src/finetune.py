"""
best_model 이어서 미세조정 (fine-tuning)

배경:
    5M 본학습은 1.8M에서 정점(eval 849.9)을 찍은 뒤 LR 고정(3e-4)으로 인해
    정책이 진동·붕괴하여 5M 시점엔 187까지 하락했다. best_model은 그 정점을
    EvalCallback이 저장한 것.

전략:
    - 1.8M best_model(eval 855)에서 출발
    - 낮은 LR을 0으로 선형 감쇠 → peak-and-collapse 방지
    - 새 best는 models/best_model_ft/ 에 저장하여 원본 best_model 보존
    - target_kl=0.02 유지 (KL 폭발 방지)

    ※ 결과 기록: 이 미세조정은 최고 804.7로 원본 855를 넘지 못해 폐기됨.
       855가 좁고 날카로운 최적점이라 PPO 업데이트로 한 번 밀리면 복귀 불가.

실행 (프로젝트 루트에서):
    python src/finetune.py --timesteps 2000000 --lr 5e-5
"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList,
)
from stable_baselines3.common.utils import get_schedule_fn

import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",    type=str,   default=str(common.DEFAULT_MODEL))
    parser.add_argument("--n-envs",    type=int,   default=8)
    parser.add_argument("--timesteps", type=int,   default=2_000_000)
    parser.add_argument("--lr",        type=float, default=5e-5)
    parser.add_argument("--device",    type=str,   default="cuda")
    args = parser.parse_args()

    device = common.get_device(args.device)
    print(f"resume : {args.resume}")
    print(f"lr     : {args.lr} → 0 (linear decay)")
    print(f"steps  : {args.timesteps} | n_envs : {args.n_envs}")

    # ── 환경 ──────────────────────────────────────────────────────────
    train_env = common.make_carracing_vec(
        args.n_envs, monitor_dir=common.RUNS_DIR / "logs_ft")
    eval_env = common.make_carracing_vec(4)

    # ── 콜백 ──────────────────────────────────────────────────────────
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(common.MODELS_DIR / "best_model_ft"),  # 원본 보존
        log_path=str(common.RUNS_DIR / "logs_ft"),
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(200_000 // args.n_envs, 1),
        save_path=str(common.RUNS_DIR / "checkpoints_ft"),
        name_prefix="ppo_ft",
        verbose=1,
    )

    # ── 모델 로드 + LR 스케줄 덮어쓰기 ────────────────────────────────
    model = PPO.load(
        args.resume, env=train_env, device=device,
        tensorboard_log=str(common.RUNS_DIR / "tb_logs"),
    )
    model.learning_rate = common.linear_schedule(args.lr)
    model.lr_schedule = get_schedule_fn(model.learning_rate)
    model.target_kl = 0.02

    print(f"로드 완료. LR check: "
          f"1.0 → {model.lr_schedule(1.0):.2e}, "
          f"0.5 → {model.lr_schedule(0.5):.2e}, "
          f"0.0 → {model.lr_schedule(0.0):.2e}")

    # ── 학습 ──────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([eval_cb, checkpoint_cb]),
        reset_num_timesteps=True,
        tb_log_name="PPO_CarRacing_FT",
        progress_bar=True,
    )

    model.save(str(common.MODELS_DIR / "ppo_carracing_ft_final"))
    print(f"저장 완료: {common.MODELS_DIR / 'ppo_carracing_ft_final'}.zip")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
