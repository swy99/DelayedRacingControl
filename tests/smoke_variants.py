"""
변형 인프라 스모크 점검 — 5일 학습 전 빠른 검증.

1) 의존성 import (sb3_contrib RecurrentPPO, gymnasium_delay)
2) AugmentDelayWrapper obs space/shape (ActStack, Both) — 단일 env, 멀티프로세싱 X
3) [--vec]    make_arch_env로 각 arch VecEnv 생성 + 1 step (SubprocVecEnv 경로)
4) [--policy] 각 arch 정책 생성 + learn(짧게) — Dict/Recurrent/VecTranspose 경로 검증

실행 (프로젝트 루트에서):
    python tests/smoke_variants.py            # 1+2 (빠름)
    python tests/smoke_variants.py --vec --policy
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import gymnasium as gym


def check_imports():
    import stable_baselines3 as sb3
    import sb3_contrib
    from sb3_contrib import RecurrentPPO            # noqa: F401
    from gymnasium_delay import ObsActDelayWrapper  # noqa: F401
    print(f"[ok] sb3={sb3.__version__} sb3_contrib={sb3_contrib.__version__}")


def check_wrapper():
    from wrappers import AugmentDelayWrapper

    # ActStack: n_obs=1, n_act=20  → image(96,96,3) + act_hist(60)
    e = AugmentDelayWrapper(gym.make("CarRacing-v3"), delay=10, n_obs=1, n_act=20)
    obs, _ = e.reset(seed=0)
    assert obs["image"].shape == (96, 96, 3), obs["image"].shape
    assert obs["act_hist"].shape == (60,), obs["act_hist"].shape
    a = e.action_space.sample()
    obs, *_ = e.step(a)
    assert obs["act_hist"].shape == (60,)
    assert np.allclose(obs["act_hist"][-3:], np.asarray(a, np.float32)), \
        "직전 emit action이 act_hist 끝에 기록돼야 함"
    e.close()
    print(f"[ok] ActStack reset/step obs={ {k: v.shape for k, v in obs.items()} }")

    # Both: n_obs=20, n_act=20  → image(96,96,60) + act_hist(60)
    e = AugmentDelayWrapper(gym.make("CarRacing-v3"), delay=10, n_obs=20, n_act=20)
    obs, _ = e.reset(seed=0)
    assert obs["image"].shape == (96, 96, 60), obs["image"].shape
    assert obs["act_hist"].shape == (60,), obs["act_hist"].shape
    e.close()
    print(f"[ok] Both reset obs={ {k: v.shape for k, v in obs.items()} }")


def check_vec():
    import common
    for arch in ("gru", "actstack", "obsstack", "both"):
        env = common.make_arch_env(2, arch, 10, seed=0)
        obs = env.reset()
        shapes = ({k: v.shape for k, v in obs.items()}
                  if isinstance(obs, dict) else obs.shape)
        env.step(np.array([env.action_space.sample() for _ in range(2)]))
        env.close()
        print(f"[ok] make_arch_env({arch}) obs={shapes}")


def check_policy():
    import common
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    for arch in ("gru", "actstack", "obsstack", "both"):
        spec = common.ARCH_SPEC[arch]
        Algo = RecurrentPPO if spec["recurrent"] else PPO
        env = common.make_arch_env(2, arch, 10, seed=0)
        model = Algo(spec["policy"], env, n_steps=64, batch_size=64,
                     device="cpu", verbose=0)
        model.learn(128)
        # predict 1회 (recurrent 상태 처리 포함)
        obs = env.reset()
        if spec["recurrent"]:
            model.predict(obs, state=None,
                          episode_start=np.ones(2, bool), deterministic=True)
        else:
            model.predict(obs, deterministic=True)
        env.close()
        print(f"[ok] {arch}: {Algo.__name__}/{spec['policy']} learn(128)+predict 통과")


if __name__ == "__main__":
    check_imports()
    check_wrapper()
    if "--vec" in sys.argv:
        check_vec()
    if "--policy" in sys.argv:
        check_policy()
    print("스모크 점검 완료")
