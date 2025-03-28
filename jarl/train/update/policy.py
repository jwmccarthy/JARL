import torch as th
from torch.optim import Adam

from jarl.data.types import LossInfo
from jarl.modules.policy import Policy
from jarl.modules.types import QFunction
from jarl.data.multi import MultiTensor
from jarl.train.optim import Optimizer, Scheduler
from jarl.train.update.base import GradientUpdate


class MaxQPolicyUpdate(GradientUpdate):

    _requires_keys = {"obs"}

    def __init__(
        self, 
        freq: int,
        policy: Policy, 
        q_func: QFunction,
        optimizer: Optimizer = Optimizer(Adam),
        scheduler: Scheduler = None,
        gamma: float = 0.99
    ) -> None:
        super().__init__(
            freq, policy, 
            optimizer=optimizer,
            scheduler=scheduler
        )
        self.policy = policy
        self.q_func = q_func
        self.gamma = gamma

    def loss(self, data: MultiTensor) -> LossInfo:
        acts = self.policy.action(data.obs)
        loss = -self.q_func(data.obs, acts).mean()
        return loss, dict(policy_loss=loss.item())


class ClippedPolicyUpdate(GradientUpdate):

    _requires_keys = {"obs", "act", "adv", "lgp", "val"}

    def __init__(
        self, 
        freq: int,
        policy: Policy, 
        optimizer: Optimizer = Optimizer(Adam),
        scheduler: Scheduler = None,
        clip: float = 0.2,
        ent_coef: float = 0.01
    ) -> None:
        super().__init__(
            freq, policy, 
            optimizer=optimizer,
            scheduler=scheduler
        )
        self.policy = policy
        self.clip = clip
        self.ent_coef = ent_coef

    def loss(self, data: MultiTensor) -> LossInfo:
        lgp = self.policy.logprob(data.obs, data.act)
        ent = self.policy.entropy(data.obs)

        lograt = lgp - data.lgp
        ratios = lograt.exp()
        with th.no_grad():
            approx_kl = ((ratios - 1) - lograt).mean().item()

        # policy loss
        norm_adv = (data.adv - data.adv.mean()) / (data.adv.std() + 1e-8)
        p_loss = -th.min(
            norm_adv * ratios,
            norm_adv * th.clamp(ratios, 1 - self.clip, 1 + self.clip)
        ).mean()

        # entropy loss
        e_loss = self.ent_coef * ent.mean()

        # total loss
        t_loss = p_loss - e_loss

        return t_loss, dict(
            policy_loss=p_loss.item(),
            entropy_loss=e_loss.item(),
            approx_kl=approx_kl
        )