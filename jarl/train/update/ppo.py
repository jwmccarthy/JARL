from typing import Set

from jarl.data.types import LossInfo
from jarl.data.core import MultiTensor

from jarl.modules.policy import Policy
from jarl.modules.operator import ValueFunction

from jarl.train.optim import Optimizer, Scheduler
from jarl.train.update.base import GradientUpdate
from jarl.train.update.policy import ClippedPolicyUpdate
from jarl.train.update.critic import MSEValueFunctionUpdate


class PPOUpdate(GradientUpdate):

    _requires_keys = {"obs", "act", "adv", "lgp", "val", "ret"}

    def __init__(
        self, 
        freq: int,
        policy: Policy, 
        critic: ValueFunction,
        optimizer: Optimizer = None,
        scheduler: Scheduler = None,
        clip: float = 0.2,
        val_coef: float = 0.5,
        ent_coef: float = 0.01,
    ) -> None:
        super().__init__(
            freq, [policy, critic], 
            optimizer=optimizer,
            scheduler=scheduler
        )
        self.policy = policy
        self.critic = critic

        # hyperparams
        self.clip = clip
        self.val_coef = val_coef
        self.ent_coef = ent_coef

        # initialize sub-updates for loss calc
        self.policy_loss = ClippedPolicyUpdate(
            freq, policy, clip=clip, ent_coef=ent_coef
        ).loss
        self.critic_loss = MSEValueFunctionUpdate(
            freq, critic, clip=clip, val_coef=val_coef
        ).loss
    
    def loss(self, data: MultiTensor) -> LossInfo:
        p_loss, p_info = self.policy_loss(data)
        v_loss, v_info = self.critic_loss(data)
        return p_loss + v_loss, p_info | v_info
