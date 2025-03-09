import torch as th
import torch.nn as nn

from typing import Self

from jarl.envs.gym import SyncEnv
from jarl.modules.encoder.core import Encoder


class CompositeNet(nn.Module):

    model: nn.Module

    def __init__(
        self,
        head: Encoder,
        body: nn.Module,
        foot: nn.Module = None
    ) -> None:
        super().__init__()
        self.head = head
        self.body = body
        self.foot = foot

    def build(self, env: SyncEnv, out_dim: int = 1) -> Self:
        self.head = self.head if self.head.built else self.head.build(env)
        self.body.build(self.head.feats, out_dim)
        self.foot = self.foot if self.foot else nn.Identity()
        self.model = nn.Sequential(self.head, self.body, self.foot)
        return self
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)