import torch as th

from typing import Set

from jarl.data.multi import MultiTensor
from jarl.modules.policy import Policy
from jarl.modules.operator import Critic
from jarl.modules.discriminator import Discriminator
from jarl.train.modify.base import DataModifier


class ComputeValues(DataModifier):

    def __init__(self, critic: Critic) -> None:
        self.critic = critic

    @property
    def requires_keys(self) -> Set[str]:
        return {"obs"} # assume default buffer input

    @property
    def produces_keys(self) -> Set[str]:
        return {"val", "next_val"}

    def __call__(self, data: MultiTensor) -> MultiTensor:
        data.set(val=self.critic(data.obs))
        data.set(next_val=self.critic(data.next_obs))
        return data
    

class ComputeLogProbs(DataModifier):

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    @property
    def requires_keys(self) -> Set[str]:
        return {"obs", "act"}

    @property
    def produces_keys(self) -> Set[str]:
        return {"lgp"}

    def __call__(self, data: MultiTensor) -> MultiTensor:
        data.set(lgp=self.policy.logprob(data.obs, data.act))
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
        return {"rew", "val", "next_val"}

    @property
    def produces_keys(self) -> Set[str]:
        return {"adv"}

    def __call__(self, data: MultiTensor) -> MultiTensor:
        don, trc = data.don, data.trc
        data.set(adv=th.zeros_like(data.rew))

        # compute TD errors
        # data.rew[trc & ~don] += data.next_val[trc & ~don]
        deltas = data.rew + self.gamma * data.next_val * ~don - data.val
        discnt = self.gamma * self.lmbda * ~don

        adv = 0
        for t in reversed(range(len(data.adv))):
            data.adv[t] = adv = deltas[t] + discnt[t] * adv

        return data


class ComputeReturns(DataModifier):

    @property
    def requires_keys(self) -> Set[str]:
        return {"val", "adv"}

    @property
    def produces_keys(self) -> Set[str]:
        return {"ret"}

    def __call__(self, data: MultiTensor) -> MultiTensor:
        data.set(ret=data.val + data.adv)
        return data
    

class DiscriminatorReward(DataModifier):

    def __init__(self, discriminator: Discriminator) -> None:
        self.discriminator = discriminator

    @property
    def requires_keys(self) -> Set[str]:
        return {"obs", "next_obs"}
    
    @property
    def produces_keys(self) -> Set[str]:
        return {"rew"}
    
    def __call__(self, data: MultiTensor) -> MultiTensor:
        data.set(raw_rew=data.rew)
        data.set(rew=-th.log(self.discriminator(data.obs, data.next_obs) + 1e-8))
        return data