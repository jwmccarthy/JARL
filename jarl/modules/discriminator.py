import torch as th
import torch.nn as nn

from typing import Self, Tuple

from jarl.envs.gym import TorchGymEnv
from jarl.modules.encoder import Encoder
from jarl.modules.composite import CompositeNet


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
    
    def forward(self, x: Tuple[th.Tensor]) -> th.Tensor:
        return self.model(x).squeeze()