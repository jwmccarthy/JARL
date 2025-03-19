import numpy as np

import gymnasium as gym
from typing import Any

from jarl.data.dict import DotDict


class EpisodeStatsEnv(gym.Wrapper):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.reward = self.length = 0

    def reset(self, seed: int = None, **kwargs) -> Any:
        self.reward = self.length = 0
        return super().reset(seed=seed, **kwargs)
    
    def step(self, act: np.ndarray):
        obs, rew, trm, trc, _ = super().step(act)
        self.reward += rew
        self.length += 1
        info = DotDict()
        if np.logical_or(trm, trc):
            info.update(rew=self.reward, len=self.length)
        return obs, rew, trm, trc, info