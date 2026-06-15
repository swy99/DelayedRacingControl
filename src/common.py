"""
공통 모듈 — 경로 상수, device 감지, LR 스케줄, 환경 빌더, SEM 수렴 평가 루프.

train.py / finetune.py / train_variant.py / eval_converge.py / eval_variant.py /
make_gif.py 등에서 공유한다. 경로는 __file__ 기준 절대경로라 어느 CWD에서
실행해도 동작한다.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, VecFrameStack, VecTransposeImage,
)
from gymnasium_delay import ObsActDelayWrapper
from wrappers import AugmentDelayWrapper

# ── 경로 ──────────────────────────────────────────────────────────────
SRC_DIR       = Path(__file__).resolve().parent
PROJECT_ROOT  = SRC_DIR.parent
MODELS_DIR    = PROJECT_ROOT / "models"
RESULTS_DIR   = PROJECT_ROOT / "results"
RUNS_DIR      = PROJECT_ROOT / "runs"
DEFAULT_MODEL = MODELS_DIR / "best_model" / "best_model"

for _d in (MODELS_DIR, RESULTS_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ENV_ID = "CarRacing-v3"

# ── 아키텍처 스펙 (train_variant·eval_variant 공유) ───────────────────
# n_obs/n_act>0 → AugmentDelayWrapper(Dict obs), n_stack>1 → VecFrameStack.
# naive는 855(DEFAULT_MODEL) 재사용 — 학습 대상 아님(참고용으로 스펙만 포함).
ARCH_SPEC = {
    "naive":    {"algo": "ppo",           "policy": "CnnPolicy",        "n_obs": 1,  "n_act": 0,  "n_stack": 1,  "recurrent": False},
    "gru":      {"algo": "recurrent_ppo", "policy": "CnnLstmPolicy",    "n_obs": 1,  "n_act": 0,  "n_stack": 1,  "recurrent": True},
    "actstack": {"algo": "ppo",           "policy": "MultiInputPolicy", "n_obs": 1,  "n_act": 20, "n_stack": 1,  "recurrent": False},
    "obsstack": {"algo": "ppo",           "policy": "CnnPolicy",        "n_obs": 1,  "n_act": 0,  "n_stack": 20, "recurrent": False},
    "both":     {"algo": "ppo",           "policy": "MultiInputPolicy", "n_obs": 20, "n_act": 20, "n_stack": 1,  "recurrent": False},
}


# ── device ────────────────────────────────────────────────────────────
def get_device(pref: str = "cuda") -> str:
    dev = pref if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"device : {dev}  ({torch.cuda.get_device_name(0)})")
    else:
        print(f"device : {dev}")
    return dev


# ── LR 스케줄 ─────────────────────────────────────────────────────────
def linear_schedule(initial_value: float):
    """progress_remaining(1→0)에 비례해 LR을 initial_value→0으로 선형 감쇠."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# ── CarRacing 벡터 환경 ───────────────────────────────────────────────
def make_carracing_vec(n_envs, *, delay=0, n_stack=1, n_obs=1, n_act=0,
                       monitor_dir=None, seed=None, render_mode=None, transpose=False):
    """CarRacing 병렬 벡터 환경을 만든다.

    n_act>0      → AugmentDelayWrapper(delay, n_obs, n_act): 딜레이 + obs를
                   Dict(image[채널 n_obs배], act_hist[n_act개])로 augment (ActStack/Both)
    elif delay>0 → ObsActDelayWrapper(act_delay=delay): 액션 딜레이만 (naive/GRU)
    n_stack>1    → VecFrameStack: 관측 n_stack 프레임 채널 스택 (ObsStack)
    transpose    → VecTransposeImage: CHW 변환 (eval 수동 predict 루프용).
                   학습 시에는 SB3가 자동 적용하므로 False로 둔다.
    """
    if n_act and n_act > 0:
        wrapper_class = AugmentDelayWrapper
        wrapper_kwargs = {"delay": int(delay), "n_obs": int(n_obs), "n_act": int(n_act)}
    elif delay and delay > 0:
        wrapper_class = ObsActDelayWrapper
        wrapper_kwargs = {"act_delay": int(delay)}
    else:
        wrapper_class, wrapper_kwargs = None, None

    venv = make_vec_env(
        ENV_ID, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv,
        monitor_dir=str(monitor_dir) if monitor_dir else None,
        env_kwargs={"render_mode": render_mode} if render_mode else {},
        wrapper_class=wrapper_class, wrapper_kwargs=wrapper_kwargs,
    )
    if n_stack and n_stack > 1:
        venv = VecFrameStack(venv, n_stack=n_stack)
    if transpose:
        venv = VecTransposeImage(venv)
    return venv


def make_arch_env(n_envs, arch, delay, *, monitor_dir=None, seed=None, transpose=False):
    """ARCH_SPEC[arch] 구성으로 delay 환경을 만든다 (train_variant·eval_variant 공유)."""
    spec = ARCH_SPEC[arch]
    return make_carracing_vec(
        n_envs, delay=delay, n_stack=spec["n_stack"],
        n_obs=spec["n_obs"], n_act=spec["n_act"],
        monitor_dir=monitor_dir, seed=seed, transpose=transpose)


def variant_dir(base, arch, init, delay):
    """변형 실험 셀의 경로를 만든다: base/variants/<arch>/<init>/d<delay>/"""
    d = base / "variants" / arch / init / f"d{delay}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── SEM 기반 수렴 평가 (eval_converge / eval_variant 공유) ────────────
def collect_until_converged(model, env, n_envs, *, recurrent=False,
                            min_ep=30, max_ep=400, sem_target=5.0, patience=16,
                            label=None, verbose=True):
    """env에서 SEM(std/√n)이 sem_target 미만으로 patience 연속 유지될 때까지
    에피소드 보상을 수집한다. recurrent=True면 lstm_states를 predict에 넘긴다.

    env는 transpose=True(CHW)로 만든 VecEnv여야 한다. label은 로그 표기용.
    Returns: (rewards, history[(n, sem)], converged_at|None)
    """
    obs = env.reset()
    ep_rewards = np.zeros(n_envs, dtype=np.float64)
    rewards: list[float] = []
    history: list[tuple[int, float]] = []
    streak, converged_at = 0, None
    lstm_states = None
    ep_starts = np.ones(n_envs, dtype=bool)

    while len(rewards) < max_ep:
        if recurrent:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=ep_starts, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, r, dones, infos = env.step(action)
        ep_starts = dones
        ep_rewards += r

        for i in range(n_envs):
            if dones[i]:
                rewards.append(float(ep_rewards[i]))
                ep_rewards[i] = 0.0
                n = len(rewards)
                if n >= 2:
                    sem = float(np.std(rewards, ddof=1) / np.sqrt(n))
                    history.append((n, sem))
                    if n >= min_ep and sem < sem_target:
                        streak += 1
                        if streak >= patience and converged_at is None:
                            converged_at = n
                    else:
                        streak = 0
                if verbose and (n % 8 == 0 or converged_at == n):
                    m = float(np.mean(rewards))
                    s = float(np.std(rewards, ddof=1)) if n >= 2 else 0.0
                    pre = f"{label} | " if label else ""
                    flag = "  ← 수렴" if converged_at == n else ""
                    print(f"  {pre}n={n:>3} | mean={m:>8.1f} | "
                          f"std={s:>6.1f} | SEM={s/np.sqrt(n):>5.2f}{flag}")

        # 수렴 판정 후 patience만큼 더 모았으면 종료
        if converged_at is not None and len(rewards) >= converged_at + patience:
            break

    env.close()
    return rewards, history, converged_at


def summarize(rewards, converged_at, **extra):
    """보상 리스트를 통계 dict로 요약. extra는 그대로 병합(예: delay=, cell=)."""
    arr = np.array(rewards)
    n = len(arr)
    std = float(arr.std(ddof=1)) if n >= 2 else 0.0
    sem = std / np.sqrt(n) if n else 0.0
    out = {
        "n": n,
        "mean": float(arr.mean()) if n else 0.0,
        "std": std,
        "sem": sem,
        "ci95_halfwidth": 1.96 * sem,
        "min": float(arr.min()) if n else 0.0,
        "max": float(arr.max()) if n else 0.0,
        "median": float(np.median(arr)) if n else 0.0,
        "fail_rate": float((arr < 0).mean()) if n else 0.0,
        "converged_at": converged_at,
        "converged": converged_at is not None,
        "rewards": rewards,
    }
    out.update(extra)
    return out
