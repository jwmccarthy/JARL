import torch as th
import torch.nn.functional as F

from typing import Set

from jarl.data.types import LossInfo
from jarl.modules.policy import Policy
from jarl.data.multi import MultiTensor
from jarl.train.optimizer import Optimizer
from jarl.train.update.base import GradientUpdate


class ClippedPolicyUpdate(GradientUpdate):

    def __init__(
        self, 
        freq: int,
        policy: Policy, 
        optimizer: Optimizer,
        epsilon: float = 0.2,
        ent_coef: float = 0.01
    ) -> None:
        super().__init__(freq, optimizer, policy)
        self.policy = policy
        self.epsilon = epsilon
        self.ent_coef = ent_coef

    @property
    def requires_keys(self) -> Set[str]:
        return {"adv", "lgp", "val"}

    def loss(self, data: MultiTensor) -> LossInfo:
        lgp = self.policy.logprob(data.obs, data.act)
        ent = self.policy.entropy(data.obs)

        with th.no_grad():
            approx_kl = (lgp - data.lgp).mean().item()

        # policy loss
        norm_adv = F.normalize(data.adv)
        ratios = th.exp(lgp - data.lgp)
        p_loss = -th.min(
            ratios * norm_adv,
            th.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * norm_adv
        )

        # entropy loss
        e_loss = -ent.mean()

        # total loss
        t_loss = p_loss + self.ent_coef * e_loss

        return t_loss, dict(
            policy_loss=p_loss.item(),
            entropy_loss=e_loss.item(),
            approx_kl=approx_kl
        )