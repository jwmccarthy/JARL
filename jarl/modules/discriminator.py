import torch as th
import torch.nn as nn

from typing import Self, Tuple

from jarl.envs.gym import TorchGymEnv
from jarl.modules.encoder import Encoder
from jarl.modules.base import CompositeNet


class Discriminator(CompositeNet):

    def __init__(
        self,
        head: Encoder,
        body: nn.Module,
        foot: nn.Module = None
    ) -> None:
        super().__init__(head, body, foot)

    def build(self, env: TorchGymEnv) -> Self:
        return super().build(env)
    
    def forward(self, x: th.Tensor | Tuple[th.Tensor]) -> th.Tensor:
        return super().forward(x).squeeze()