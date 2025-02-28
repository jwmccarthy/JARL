import torch as th

from typing import Set

from jarl.data.types import LossInfo
from jarl.data.core import MultiTensor
from jarl.modules.operator import Critic
from jarl.train.optim import Optimizer, Scheduler
from jarl.train.update.base import GradientUpdate


class MSECriticUpdate(GradientUpdate):

    _requires_keys = {"obs", "ret", "val"}

    def __init__(
        self, 
        freq: int, 
        critic: Critic,
        optimizer: Optimizer = None, 
        scheduler: Scheduler = None,
        clip: float = None,
        val_coef: float = 0.5
    ) -> None:
        super().__init__(
            freq, critic, 
            optimizer=optimizer,
            scheduler=scheduler
        )
        self.critic = critic
        self.clip = clip
        self.val_coef = val_coef

    def loss(self, data: MultiTensor) -> LossInfo:
        val = self.critic(data.obs)
        loss = (val - data.ret).pow(2)
        if self.clip:
            val_clip = data.val + th.clamp(
                val - data.val, -self.clip, self.clip)
            loss_clip = (val_clip - data.ret).pow(2)
            loss = 0.5 * th.max(loss, loss_clip)
        loss = self.val_coef * loss.mean()
        return loss, dict(critic_loss=loss.item())