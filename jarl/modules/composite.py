import torch as th
import torch.nn as nn

from typing import Self

from jarl.envs.gym import TorchGymEnv
from jarl.modules.encoder import Encoder


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

    def build(self, env: TorchGymEnv, out_dim: int = 1) -> Self:
        self.head.build(env.obs_space)
        self.body.build(self.head.feats, out_dim)
        self.foot = self.foot if self.foot else nn.Identity()
        self.model = nn.Sequential(self.head, self.body, self.foot)
        return self
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)