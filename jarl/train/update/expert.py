import torch.nn as nn
from torch.nn.functional import binary_cross_entropy

from typing import Set

from jarl.data.types import LossInfo
from jarl.data.core import MultiTensor
from jarl.train.optim import Optimizer
from jarl.train.update.base import GradientUpdate


class GAIFOUpdate(GradientUpdate):

    def __init__(
        self, 
        freq: int, 
        discrim: nn.Module,
        optimizer: Optimizer = None
    ) -> None:
        super().__init__(freq, discrim, optimizer=optimizer)
        self.discrim = discrim

    @property
    def requires_keys(self) -> Set[str]:
        return {"obs", "nxt", "lbl"}
        
    @property
    def truncate_envs(self) -> bool:
        return True
    
    def loss(self, data: MultiTensor) -> LossInfo:
        prob = self.discrim((data.obs, data.nxt))
        loss = binary_cross_entropy(prob, data.lbl)
        return loss, dict(bce_loss=loss.item())