import numpy as np
import torch as th

from typing import Self, Any

from jarl.envs.env import TorchEnv, SyncEnv


class EpisodicLifeWrapper(TorchEnv):

    def __init__(self, env: TorchEnv) -> None:
        super().__init__(env.env, env.device)
        self.lives = 0
        self.real_done = True

    def step(self, act: np.ndarray) -> th.Tensor:
        *exp, don, nxt, info = super().step(act)
        self.real_done = don
        lives = self.env.unwrapped.ale.lives()
        don = True if 0 < lives < self.lives else don
        self.lives = lives
        return *exp, don, nxt, info
    
    def reset(self) -> th.Tensor:
        if self.real_done:
            obs = super().reset()
        else:
            obs, _, _, don, *_ = super().step(np.array(0))
            obs = super().reset() if don else obs
        self.lives = self.env.unwrapped.ale.lives()
        return obs