import torch as th
import torch.nn as nn

from typing import Self, Tuple
from abc import ABC, abstractmethod

from jarl.envs.gym import TorchGymEnv
from jarl.modules.encoder.base import Encoder


class FlattenEncoder(Encoder):
    
    def build(self, env: TorchGymEnv) -> Self:
        self.feats = env.obs_space.flat_dim
        return self
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.flatten(x, -1)
    

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