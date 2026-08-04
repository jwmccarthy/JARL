import torch as th
import torch.nn as nn

from typing import Self

from jarl.envs.gym import SyncGymEnv
from jarl.modules.encoder.core import Encoder


class CompositeNet(nn.Module):

    model: nn.Module

    def __init__(
        self,
        foot: Encoder,
        body: nn.Module,
        head: nn.Module = None
    ) -> None:
        super().__init__()
        self.foot = foot
        self.body = body
        self.head = head
        self.device = "cpu"

    def to(self, device: str) -> Self:
        self.device = device
        return super().to(device)

    def build(self, env: SyncGymEnv, out_dim: int = 1) -> Self:
        self.foot = self.foot if self.foot.built else self.foot.build(env)
        self.body.build(self.foot.feats, out_dim)
        self.head = self.head if self.head else nn.Identity()
        self.model = nn.Sequential(self.foot, self.body, self.head)
        return self
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)
