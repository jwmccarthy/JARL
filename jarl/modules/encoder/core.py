import torch as th
import torch.nn as nn

from typing import Self, Tuple

from jarl.envs.vec import TorchGymEnv
from jarl.modules.encoder.base import Encoder


class FlattenEncoder(Encoder):
    
    def build(self, env: TorchGymEnv) -> Self:
        self.start_dim = -len(env.obs_space.shape)
        self.feats = env.obs_space.flat_dim
        return self
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.flatten(x, start_dim=self.start_dim)
    

class StackObsEncoder(Encoder):

    def build(self, env: TorchGymEnv) -> Self:
        self.feats = env.obs_space.flat_dim * 2
        return self
    
    def forward(self, x: Tuple[th.Tensor, ...]) -> th.Tensor:
        return th.cat(x, -1)
    

class StackObsActEncoder(Encoder):

    def build(self, env: TorchGymEnv) -> Self:
        self.feats = env.obs_space.flat_dim + env.act_space.numel
        return self
    
    def forward(self, x: Tuple[th.Tensor, ...]) -> th.Tensor:
        return th.cat(x, -1)