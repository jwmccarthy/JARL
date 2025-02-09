import torch as th
import torch.nn as nn
from torch import Tensor

from typing import Self, Tuple
from abc import ABC, abstractmethod

from jarl.envs.gym import TorchGymEnv


class Encoder(nn.Module, ABC):

    feats: int

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def build(self, env: TorchGymEnv) -> Self:
        ...
        

class FlattenEncoder(Encoder):
    
    def build(self, env: TorchGymEnv) -> Self:
        self.feats = env.obs_space.flat_dim
        return self
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.flatten(x, -1)