import torch as th
import torch.nn.functional as F

from typing import Set

from jarl.data.types import LossInfo
from jarl.data.core import MultiTensor
from jarl.modules.policy import Policy
from jarl.modules.types import QFunction
from jarl.modules.operator import ValueFunction
from jarl.train.optim import Optimizer, Scheduler
from jarl.train.update.base import GradientUpdate


class MSEValueFunctionUpdate(GradientUpdate):

    _requires_keys = {"obs", "ret", "val"}

    def __init__(
        self, 
        freq: int, 
        critic: ValueFunction,
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
    

class MSBEUpdate(GradientUpdate):

    _requires_keys = {"obs", "act", "rew", "nxt", "don", "trc"}

    def __init__(
        self,
        freq: int,
        q_func: QFunction,
        q_targ: QFunction,
        p_targ: Policy,
        optimizer: Optimizer = None,
        scheduler: Scheduler = None,
        gamma: float = 0.99
    ) -> None:
        super().__init__(
            freq, q_func, 
            optimizer=optimizer,
            scheduler=scheduler
        )
        self.q_func = q_func
        self.q_targ = q_targ
        self.p_targ = p_targ
        self.gamma = gamma

    def loss(self, data: MultiTensor) -> LossInfo:
        nontrm = (~data.don | data.trc).float()
        target = data.rew + self.gamma * nontrm \
                          * self.q_targ(data.nxt, self.p_targ(data.nxt))
        loss = F.mse_loss(self.q_func(data.obs, data.act), target)
        return loss, dict(q_loss=loss.item())