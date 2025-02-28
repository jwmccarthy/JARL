import torch.nn as nn
from torch.nn.functional import binary_cross_entropy

from typing import Set

from jarl.data.types import LossInfo
from jarl.data.core import MultiTensor
from jarl.train.optim import Optimizer, Scheduler
from jarl.train.update.base import GradientUpdate


class GAIFOUpdate(GradientUpdate):

    _requires_keys = {"obs", "nxt", "lbl"}

    def __init__(
        self, 
        freq: int, 
        discrim: nn.Module,
        optimizer: Optimizer = None,
        scheduler: Scheduler = None,
    ) -> None:
        super().__init__(
            freq, discrim, 
            optimizer=optimizer,
            scheduler=scheduler
        )
        self.discrim = discrim
        
    def loss(self, data: MultiTensor) -> LossInfo:
        prob = self.discrim((data.obs, data.nxt))
        loss = binary_cross_entropy(prob, data.lbl)
        return loss, dict(bce_loss=loss.item())