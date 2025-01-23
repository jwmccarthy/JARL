import torch as th
import torch.nn as nn
from torch import Tensor

from typing import Self, Callable
from abc import ABC, abstractmethod

from jarl.envs.space import TensorSpace


class Encoder(nn.Module, ABC):

    feats: int
    model: nn.Module

    def __init__(
        self, 
        transform: Callable[[Tensor], Tensor] = None
    ) -> None:
        super().__init__()
        self.transform = transform  # custom preprocessor

    @abstractmethod
    def build(self, obs_space: TensorSpace) -> Self:
        ...

    def forward(self, obs: th.Tensor) -> th.Tensor:
        if self.transform:
            obs = self.transform(obs)
        return self.model(obs)


class FlattenEncoder(Encoder):
    
    def __init__(
        self,
        transform: Callable[[Tensor], Tensor] = None
    ) -> None:
        super().__init__(transform)
    
    def build(self, obs_space: TensorSpace) -> Self:
        self.feats = obs_space.flat_dim
        self.model = nn.Flatten(start_dim=-len(obs_space.shape))
        return self