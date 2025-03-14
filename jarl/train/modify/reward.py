import torch as th

from jarl.data.core import MultiTensor
from jarl.train.modify.base import DataModifier


class SignRewards(DataModifier):

    _requires_keys = {"rew"}
    _produces_keys = {"rew"}

    @th.no_grad()
    def __call__(self, data: MultiTensor) -> MultiTensor:
        data.rew = data.rew.sign()
        return data
    

class NormalizeRewards(DataModifier):

    _requires_keys = {"rew", "don", "trc"}
    _produces_keys = {"rew"}

    def __init__(self, gamma: float = 0.99) -> None:
        super().__init__()
        self.gamma = gamma

    @th.no_grad()
    def __call__(self, data: MultiTensor) -> MultiTensor:
        nondon = 1 - data.don.float()
        nontrm = (~data.don | data.trc).float()
        stddev = th.zeros_like(data.rew)
        reward = th.zeros_like(data.rew)

        # discount rew & calculate moving mean
        rew = mean = std = 0
        discnt = self.gamma * nondon
        for i in range(len(data.rew)):
            m, n = i + 1, i + 2
            reward[i] = rew = data.rew[i] + discnt[i] * rew

            # moving mean & std
            delta = rew - mean
            mean = mean + 1 / n * delta if nondon[i] else rew
            stddev[i] = std = (n * std + th.square(delta) * m / n) / n

        # normalize reward
        data.rew = reward / stddev

        return data