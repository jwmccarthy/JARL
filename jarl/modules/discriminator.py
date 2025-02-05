import torch as th
import torch.nn as nn

from typing import Self

from jarl.envs.gym import TorchGymEnv
from jarl.modules.encoder import Encoder
from jarl.modules.composite import CompositeNet


class Discriminator(CompositeNet):

    def __init__(
        self,
        head: Encoder,
        body: nn.Module
    ) -> None:
        super().__init__(head, body)

    def build(self, env: TorchGymEnv) -> Self:
        self.head = self.head.build(env.obs_space)
        self.body.build(self.head.feats * 2, 1)
        self.body.model.append(nn.Sigmoid())
        return self

    def forward(self, obs_pair: th.Tensor) -> th.Tensor:
        return self.body(obs_pair).squeeze()