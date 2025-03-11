import torch as th
import torch.nn as nn

from typing import Self

from jarl.envs.gym import SyncEnv
from jarl.modules.encoder.core import Encoder
from jarl.modules.base import CompositeNet


class Critic(CompositeNet):

    model: nn.Module

    def __init__(
        self, 
        head: Encoder, 
        body: nn.Module,
        foot: nn.Module = None
    ) -> None:
        super().__init__(head, body, foot)

    def build(self, env: SyncEnv) -> Self:
        return super().build(env)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(x).squeeze(-1)
    

class QFunction(Critic):

    model: nn.Module

    def __init__(
        self, 
        head: Encoder, 
        body: nn.Module,
        foot: nn.Module = None,
        acts: bool = True
    ) -> None:
        super().__init__(head, body, foot)
        # if acts is true, put ConcatEncoder as top of head
        # and allow composition of encoders

    def build(self, env: SyncEnv) -> Self:
        return super().build(env)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(x).squeeze(-1)