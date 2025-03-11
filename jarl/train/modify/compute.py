import torch as th

from typing import Set

from jarl.data.core import MultiTensor
from jarl.modules.policy import Policy
from jarl.modules.operator import Critic
from jarl.train.modify.base import DataModifier


class ComputeValues(DataModifier):

    _requires_keys = {"obs", "nxt"}
    _produces_keys = {"val", "next_val"}

    def __init__(self, critic: Critic) -> None:
        self.critic = critic

    @th.no_grad()
    def __call__(self, data: MultiTensor) -> MultiTensor:
        data.val = self.critic(data.obs)
        data.next_val = self.critic(data.nxt)
        return data
    

class ComputeLogProbs(DataModifier):

    _requires_keys = {"obs", "act"}
    _produces_keys = {"lgp"}

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    @th.no_grad()
    def __call__(self, data: MultiTensor) -> MultiTensor:
        data.lgp = self.policy.logprob(data.obs, data.act)
        return data
    

class ComputeAdvantages(DataModifier):

    _requires_keys = {"rew", "val", "next_val"}
    _produces_keys = {"adv"}

    def __init__(
        self, 
        gamma: float = 0.99, 
        lmbda: float = 0.95
    ) -> None:
        self.gamma = gamma
        self.lmbda = lmbda

    @th.no_grad()
    def __call__(self, data: MultiTensor) -> MultiTensor:
        data.trc[-1] = ~data.don[-1]
        not_done = 1 - data.don.float()
        non_term = (~data.don | data.trc).float()
        data.adv = th.zeros_like(data.rew)

        # compute TD errors
        deltas = data.rew + self.gamma * data.next_val * non_term - data.val

        # vectorize discount factor
        discnt = self.gamma * self.lmbda * not_done

        adv = 0
        for t in reversed(range(len(data.adv))):
            data.adv[t] = adv = deltas[t] + discnt[t] * adv

        return data


class ComputeReturns(DataModifier):

    _requires_keys = {"val", "adv"}
    _produces_keys = {"ret"}

    @th.no_grad()
    def __call__(self, data: MultiTensor) -> MultiTensor:
        data.ret = data.val + data.adv
        return data