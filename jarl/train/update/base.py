import torch as th
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Any, Dict, Set, List

from jarl.data.types import LossInfo
from jarl.data.multi import MultiTensor
from jarl.train.optimizer import Optimizer


class ModuleUpdate(ABC):

    def __init__(self, freq: int) -> None:
        self.freq = freq

    @property
    @abstractmethod
    def requires_keys(self) -> Set[str]:
        ...

    @abstractmethod
    def __call__(self, data: MultiTensor) -> Dict[str, Any]:
        ...

    def ready(self, t: int) -> bool:
        return t % self.freq == 0
    

class GradientUpdate(ModuleUpdate, ABC):

    def __init__(
        self, 
        freq: int, 
        optimizer: Optimizer,
        modules: nn.Module | List[nn.Module]
    ) -> None:
        super().__init__(freq)
        if isinstance(modules, nn.Module):
            modules = [modules]
        self.optimizer = optimizer.build(*modules)

    @abstractmethod
    def loss(self, data: MultiTensor) -> LossInfo:
        ...

    def __call__(self, data: MultiTensor) -> Dict[str, Any]:
        loss, info = self.loss(data)
        self.optimizer.update(loss)
        return info