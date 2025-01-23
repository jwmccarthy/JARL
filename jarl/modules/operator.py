import torch as th
import torch.nn as nn

from typing import Self

from jarl.envs.gym import TorchGymEnv
from jarl.modules.encoder import Encoder
from jarl.modules.network import CompositeNet


class Critic(CompositeNet):

    def __init__(
        self, 
        head: Encoder, 
        body: nn.Module
    ) -> None:
        super().__init__(head, body)

    def build(self, env: TorchGymEnv) -> Self:
        return super().build(env, 1)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return super().forward(obs).squeeze()