import torch.nn as nn
import torch.nn.functional as F

from typing import Set

from jarl.data.types import LossInfo
from jarl.data.multi import MultiTensor
from jarl.train.optim import Optimizer
from jarl.train.update.base import GradientUpdate


class CrossEntropyUpdate(GradientUpdate):
    
    def __init__(
        self, 
        freq: int, 
        model: nn.Module,
        optimizer: Optimizer = None
    ) -> None:
        super().__init__(freq, model, optimizer=optimizer)
        self.model = model

    @property
    def requires_keys(self) -> Set[str]:
        return {"obs", "next_obs", "lbl"}

    def loss(self, data: MultiTensor) -> LossInfo:
        logits = self.model(data.obs, data.next_obs)
        loss = F.binary_cross_entropy_with_logits(logits, data.lbl)
        return loss, dict(x_entropy=loss.item())