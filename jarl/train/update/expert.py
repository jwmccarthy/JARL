import torch.nn as nn
from torch.optim import Optimizer, Adam
from torch.nn.functional import binary_cross_entropy

from typing import Dict, Any

from jarl.data.multi import MultiTensor
from jarl.train.update.base import GradientUpdate
from jarl.data.types import LossInfo, SchedulerFunc


class GAIFOUpdate(GradientUpdate):

    _requires_keys = {"obs", "nxt", "lbl"}

    def __init__(
        self, 
        freq: int, 
        discrim: nn.Module,
        optimizer: Optimizer = Adam,
        scheduler: SchedulerFunc = None,
        grad_norm: float = None,
        **op_kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(
            freq, discrim, 
            optimizer=optimizer,
            scheduler=scheduler,
            grad_norm=grad_norm,
            **op_kwargs
        )
        self.discrim = discrim
        
    def loss(self, data: MultiTensor) -> LossInfo:
        prob = self.discrim((data.obs, data.nxt))
        loss = binary_cross_entropy(prob, data.lbl)
        return loss, dict(bce_loss=loss.item())