from typing import Set

from jarl.data.types import LossInfo
from jarl.data.multi import MultiTensor

from jarl.modules.policy import Policy
from jarl.modules.operator import Critic

from jarl.train.optim import Optimizer
from jarl.train.update.base import GradientUpdate
from jarl.train.update.policy import ClippedPolicyUpdate
from jarl.train.update.critic import MSECriticUpdate


class PPOUpdate(GradientUpdate):

    def __init__(
        self, 
        freq: int,
        policy: Policy, 
        critic: Critic,
        optimizer: Optimizer = None,
        clip: float = 0.2,
        val_coef: float = 0.5,
        ent_coef: float = 0.01,
    ) -> None:
        super().__init__(
            freq, [policy, critic], optimizer=optimizer
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
        self.critic_loss = MSECriticUpdate(
            freq, critic, val_coef=val_coef
        ).loss

    @property
    def requires_keys(self) -> Set[str]:
        return {"adv", "lgp", "val", "ret"}
    
    def loss(self, data: MultiTensor) -> LossInfo:
        p_loss, p_info = self.policy_loss(data)
        v_loss, v_info = self.critic_loss(data)
        return p_loss + v_loss, p_info | v_info
