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
    

class NormalizedRewards(DataModifier):

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
        rew, mean, squares = 0, 0
        discnt = self.gamma * nontrm
        for i in reversed(range(len(data.rew))):
            n = len(data.rew) - i
            reward[i] = rew = data.rew[i] + discnt[i] * rew

            # moving mean & std
            mean = mean * (n - 1) / n + mean / n if nondon[i] else rew
            squares = th.pow(rew, 2) + squares * nondon[i]
            stddev[i] = (squares - th.pow(mean, 2) + 1e-8).sqrt()

        # normalize reward
        reward /= stddev