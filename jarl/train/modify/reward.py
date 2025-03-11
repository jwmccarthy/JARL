from jarl.data.core import MultiTensor
from jarl.train.modify.base import DataModifier


class SignRewards(DataModifier):

    _requires_keys = {"rew"}
    _produces_keys = {"rew"}

    @th.no_grad()
    def __call__(self, data: MultiTensor) -> MultiTensor:
        data.rew = data.rew.sign()
        return data