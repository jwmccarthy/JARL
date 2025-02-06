import torch as th
import torch.nn as nn
from torch import Tensor

from typing import Self, Tuple
from abc import ABC, abstractmethod

from jarl.envs.space import TensorSpace


class Encoder(nn.Module, ABC):

    feats: int

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def build(self, space: TensorSpace) -> Self:
        ...
        

class FlattenEncoder(Encoder):
    
    def build(self, space: TensorSpace) -> Self:
        self.feats = space.flat_dim
        return self
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.flatten(x, start_dim=-1)
    

class StackObsEncoder(Encoder):

    def build(self, space: TensorSpace) -> Self:
        self.feats = space.flat_dim * 2
        return self

    def forward(self, x: Tuple[th.Tensor]) -> th.Tensor:
        return th.cat(x, dim=-1)