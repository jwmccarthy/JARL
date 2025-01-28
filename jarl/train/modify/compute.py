import torch as th

from typing import Set

from jarl.data.multi import MultiTensor
from jarl.modules.operator import Critic
from jarl.train.modify.base import DataModifier


class ComputeValues(DataModifier):

    def __init__(self, critic: Critic) -> None:
        self.critic = critic

    @property
    def requires_keys(self) -> Set[str]:
        return set()  # assume default buffer input

    @property
    def produces_keys(self) -> Set[str]:
        return {"val", "next_val"}

    @th.no_grad()
    def __call__(self, data: MultiTensor) -> MultiTensor:
        data.val = self.critic(data.obs)
        data.next_val = self.critic(data.next_obs)
        return data
    

class ComputeAdvantages(DataModifier):

    def __init__(
        self, 
        gamma: float = 0.99, 
        lmbda: float = 0.95
    ) -> None:
        self.gamma = gamma
        self.lmbda = lmbda

    @property
    def requires_keys(self) -> Set[str]:
        return {"val", "next_val"}

    @property
    def produces_keys(self) -> Set[str]:
        return {"adv"}

    def __call__(self, data: MultiTensor) -> MultiTensor:
        # don, trc = data.don, data.trc
        # data.adv = th.zeros_like(data.rew)

        # # compute TD errors
        # deltas = data.rew + self.gamma * data.next_val - data.val
        # nontrm = ~don

        # adv = 0
        # for t in reversed(range(len(deltas))):
        #     data.adv[t] = adv = deltas[t] + self.gamma * self.lmbda * adv
        # return data
        return data


class ComputeReturns(DataModifier):

    @property
    def requires_keys(self) -> Set[str]:
        return {"val", "adv"}

    @property
    def produces_keys(self) -> Set[str]:
        return {"ret"}

    def __call__(self, data: MultiTensor) -> MultiTensor:
        data.ret = data.val + data.adv
        return data