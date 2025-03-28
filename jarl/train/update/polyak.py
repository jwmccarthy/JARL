import torch as th
import torch.nn as nn

from typing import Set

from jarl.train.update.base import ModuleUpdate
from jarl.data.types import LossInfo
from jarl.data.multi import MultiTensor


class PolyakUpdate(ModuleUpdate):

    def __init__(
        self, 
        freq: int,
        source: nn.Module,
        target: nn.Module,
        tau: float = 0.005
    ) -> None:
        super().__init__(freq)
        self.source = source
        self.target = target
        self.tau = tau

    @property
    def requires_keys(self) -> Set[str]:
        return set()

    def __call__(self, data: MultiTensor) -> LossInfo:
        device = next(self.source.parameters()).device
        one = th.ones(1, requires_grad=False).to(device)
        src_params = self.source.parameters()
        trg_params = self.target.parameters()

        for src_param, trg_param in zip(src_params, trg_params):
            trg_param.data.mul_(1 - self.tau)
            trg_param.data.addcmul_(src_param.data, one, value=self.tau)

        return dict()