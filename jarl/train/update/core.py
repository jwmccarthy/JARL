import torch as th
import torch.nn as nn

from typing import Set

from jarl.data.types import LossInfo
from jarl.data.multi import MultiTensor
from jarl.train.optim import Optimizer
from jarl.train.update.base import GradientUpdate


class CrossEntropyUpdate(GradientUpdate):
    
    loss_func = nn.BCELoss()

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
        return {"pol_obs", "exp_obs"}

    def loss(self, data: MultiTensor) -> LossInfo:
        # current policy loss
        pol_prob = self.model(data.pol_obs)
        pol_loss = self.loss_func(pol_prob, th.ones_like(pol_prob))

        # expert data loss
        exp_prob = self.model(data.exp_obs)
        exp_loss = self.loss_func(exp_prob, th.zeros_like(exp_prob))

        loss = pol_loss + exp_loss

        return loss, dict(
            exp_loss=exp_loss.item(),
            pol_loss=pol_loss.item(),
            bce_loss=loss.item()
        )