"""
변형 모델용 커스텀 환경 wrapper.

AugmentDelayWrapper — 액션 딜레이(delay)를 적용하면서 관측을 augmented state로
만든다. 지연 MDP를 다시 Markov로 만드는 정석 해법(현재 state + 버퍼에 들어있는
미적용 action)을 따른다.

  - n_obs>1 : 최근 n_obs 프레임을 채널축으로 concat (속도·궤적 등 motion 정보)
  - n_act>0 : 에이전트가 직전 n_act 스텝 동안 emit한 action을 flatten

관측은 Dict({"image": (96,96,3*n_obs), "act_hist": (n_act*act_dim,)})가 되며,
SB3 MultiInputPolicy(CombinedExtractor)가 image는 CNN, act_hist는 MLP로 처리한다.
ActStack=(n_obs=1, n_act=20)·Both=(n_obs=20, n_act=20) 두 경우에만 쓰인다.
"""
from __future__ import annotations
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium_delay import ObsActDelayWrapper

__all__ = ["AugmentDelayWrapper"]


class AugmentDelayWrapper(gym.Wrapper):
    """act_delay 적용 + obs를 (최근 n_obs 프레임, 최근 n_act emit action)으로 augment."""

    def __init__(self, env: gym.Env, delay: int, n_obs: int = 1, n_act: int = 0):
        # 실제 액션 지연은 기존 ObsActDelayWrapper가 담당한다.
        env = ObsActDelayWrapper(env, act_delay=int(delay))
        super().__init__(env)
        self.delay = int(delay)
        self.n_obs = max(int(n_obs), 1)
        self.n_act = max(int(n_act), 0)

        img_space = env.observation_space            # Box(96,96,3) uint8
        h, w, c = img_space.shape
        self._act_dim = int(np.prod(env.action_space.shape))

        self._frames: deque = deque(maxlen=self.n_obs)
        self._acts: deque = deque(maxlen=max(self.n_act, 1))

        obs_spaces = {
            "image": spaces.Box(
                low=0, high=255,
                shape=(h, w, c * self.n_obs), dtype=img_space.dtype),
        }
        if self.n_act > 0:
            obs_spaces["act_hist"] = spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.n_act * self._act_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)

    def _augment(self):
        # 프레임은 오래된→최신 순으로 채널 concat.
        out = {"image": np.concatenate(list(self._frames), axis=-1)}
        if self.n_act > 0:
            out["act_hist"] = np.concatenate(list(self._acts), dtype=np.float32)
        return out

    def reset(self, **kwargs):
        img, info = self.env.reset(**kwargs)
        self._frames.clear()
        for _ in range(self.n_obs):
            self._frames.append(np.zeros_like(img))
        self._frames.append(img)                      # 최신 프레임 (maxlen이 밀어냄)
        self._acts.clear()
        for _ in range(max(self.n_act, 1)):
            self._acts.append(np.zeros(self._act_dim, dtype=np.float32))
        return self._augment(), info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.n_act > 0:
            self._acts.append(a)                      # emit한 action 기록
        img, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(img)
        return self._augment(), reward, terminated, truncated, info
