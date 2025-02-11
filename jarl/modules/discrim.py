import torch as th
import torch.nn as nn

from jarl.modules.encoder import Encoder
from jarl.modules.base import CompositeNet


class Discriminator(CompositeNet):
    
    def __init__(
        self, 
        head: Encoder, 
        body: nn.Module,
        foot: nn.Module = nn.Sigmoid()
    ) -> None:
        super().__init__(head, body, foot)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(x).squeeze()