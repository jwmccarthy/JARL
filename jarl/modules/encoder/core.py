import torch as th
import torch.nn as nn

from typing import Self, Tuple

from jarl.envs.gym import SyncGymEnv
from jarl.modules.encoder.base import Encoder


class FlattenEncoder(Encoder):
    
    def build(self, env: SyncGymEnv) -> Self:
        super().build(env)
        self.start_dim = -len(env.obs_space.shape)
        self.feats = env.obs_space.flat_dim
        return self
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.flatten(x, start_dim=self.start_dim)