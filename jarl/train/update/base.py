import torch as th
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Any, Dict, Set, List

from jarl.data.types import LossInfo
from jarl.data.core import MultiTensor
from jarl.train.optim import Optimizer


class ModuleUpdate(ABC):

    def __init__(self, freq: int) -> None:
        self.freq = freq

    @property
    @abstractmethod
    def requires_keys(self) -> Set[str]:
        ...

    @property
    @abstractmethod
    def truncate_envs(self) -> bool:
        ...

    @abstractmethod
    def __call__(self, data: MultiTensor) -> Dict[str, Any]:
        ...

    def ready(self, t: int) -> bool:
        return t != 0 and t % self.freq == 0
    

class GradientUpdate(ModuleUpdate, ABC):

    def __init__(
        self, 
        freq: int, 
        modules: nn.Module | List[nn.Module],
        optimizer: Optimizer = None
    ) -> None:
        super().__init__(freq)

        if not isinstance(modules, list):
            modules = [modules]
        self.modules = modules

        # build optimizer if given
        self.optimizer = optimizer
        if self.optimizer is not None:
            self.build(self.optimizer)

    def build(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        self.optimizer.build(self.modules)
        return self

    @abstractmethod
    def loss(self, data: MultiTensor) -> LossInfo:
        ...

    def __call__(self, data: MultiTensor) -> Dict[str, Any]:
        loss, info = self.loss(data)
        self.optimizer.update(loss)
        return info