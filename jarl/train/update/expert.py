import torch as th
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
        model: nn.Module,
        optimizer: Optimizer = None
    ) -> None:
        super().__init__(freq, model, optimizer=optimizer)
        self.model = model

    @property
    def requires_keys(self) -> Set[str]:
        return {"obs", "nxt", "exp_obs", "exp_nxt"}

    def loss(self, data: MultiTensor) -> LossInfo:
        # current policy loss
        pol_prob = self.model((data.obs, data.nxt))
        pol_loss = binary_cross_entropy(pol_prob, th.ones_like(pol_prob))

        # expert data loss
        exp_prob = self.model((data.exp_obs, data.exp_nxt))
        exp_loss = binary_cross_entropy(exp_prob, th.zeros_like(exp_prob))

        bce_loss = pol_loss + exp_loss

        return bce_loss, dict(
            exp_loss=exp_loss.item(),
            pol_loss=pol_loss.item(),
            bce_loss=bce_loss.item()
        )