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
        self.built = False

    def to(self, device: str) -> Self:
        self.device = device
        return super().to(device)

    def build(self, env: SyncGymEnv, out_dim: int = 1) -> Self:
        self.foot = self.foot if self.foot.built else self.foot.build(env)
        if hasattr(self.body, "build") and not getattr(self.body, "built", False):
            self.body.build(self.foot.feats)
        self._build_head(self.body.feats, out_dim)
        self.built = True
        return self

    def _build_head(self, in_dim: int, out_dim: int) -> None:
        if self.head is None:
            self.head = nn.Linear(in_dim, out_dim)
        elif hasattr(self.head, "build") and not getattr(self.head, "built", False):
            self.head.build(in_dim, out_dim)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.head(self.body(self.foot(x)))
