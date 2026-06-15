"""
CarRacing-v3 PPO 학습 (기본 baseline)

- SubprocVecEnv 병렬 환경 (common.make_carracing_vec)
- EvalCallback으로 수렴 추적, CheckpointCallback으로 체크포인트 저장
- TensorBoard + CSV 로그

실행 (프로젝트 루트에서):
    python src/train.py [--n-envs 8] [--timesteps 5000000]

TensorBoard:
    tensorboard --logdir runs/tb_logs
"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList,
)

import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs",    type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--device",    type=str, default="cuda")
    parser.add_argument("--resume",    type=str, default=None,
                        help="이어서 학습할 모델 경로 (.zip 제외)")
    args = parser.parse_args()

    device = common.get_device(args.device)
    print(f"n_envs : {args.n_envs}")

    # ── 환경 ──────────────────────────────────────────────────────────
    train_env = common.make_carracing_vec(
        args.n_envs, monitor_dir=common.RUNS_DIR / "logs")
    eval_env = common.make_carracing_vec(4)

    # ── 콜백 ──────────────────────────────────────────────────────────
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(common.MODELS_DIR / "best_model"),
        log_path=str(common.RUNS_DIR / "logs"),
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=str(common.RUNS_DIR / "checkpoints"),
        name_prefix="ppo_carracing",
        verbose=1,
    )

    # ── 모델 ──────────────────────────────────────────────────────────
    if args.resume:
        print(f"이어서 학습: {args.resume}")
        model = PPO.load(
            args.resume, env=train_env, device=device,
            tensorboard_log=str(common.RUNS_DIR / "tb_logs"),
        )
    else:
        model = PPO(
            "CnnPolicy", train_env,
            n_steps=512, batch_size=256, learning_rate=3e-4,
            target_kl=0.02,
            tensorboard_log=str(common.RUNS_DIR / "tb_logs"),
            device=device, verbose=1,
        )

    # ── 학습 ──────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([eval_cb, checkpoint_cb]),
        reset_num_timesteps=not bool(args.resume),
        tb_log_name="PPO_CarRacing",
        progress_bar=True,
    )

    model.save(str(common.MODELS_DIR / "ppo_carracing_final"))
    print(f"저장 완료: {common.MODELS_DIR / 'ppo_carracing_final'}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
