import torch.nn as nn
from torch.optim import Optimizer, Adam
from torch.nn.utils import clip_grad_norm_

from abc import ABC, abstractmethod
from typing import Any, Dict, Set, List

from jarl.data.multi import MultiTensor
from jarl.data.types import LossInfo, SchedulerFunc


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
        return t != 0 and t % self.freq == 0
    

class GradientUpdate(ModuleUpdate, ABC):

    _requires_keys: Set[str]

    def __init__(
        self, 
        freq: int, 
        modules: List[nn.Module],
        optimizer: Optimizer = Adam,
        scheduler: SchedulerFunc = None,
        grad_norm: float = None,
        **op_kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(freq)

        # compile distinct parameters
        self.params = nn.ParameterList(
            dict.fromkeys(p for m in modules for p in m.parameters()))
        
        self.grad_norm = grad_norm
        self.optimizer = optimizer(self.params, **op_kwargs)
        self.scheduler_func = scheduler

    def init_scheduler(self, steps: int) -> None:
        if not self.scheduler_func: return
        self.scheduler = self.scheduler_func(self.optimizer, steps)
    
    def step_scheduler(self) -> None:
        if self.scheduler: self.scheduler.step()

    @property
    def requires_keys(self) -> Set[str]:
        return self._requires_keys

    @abstractmethod
    def loss(self, data: MultiTensor) -> LossInfo:
        ...

    def __call__(self, data: MultiTensor) -> Dict[str, Any]:
        loss, info = self.loss(data)
        self.optimizer.zero_grad()
        loss.backward()

        # clip gradients
        if self.grad_norm:
            clip_grad_norm_(self.params, self.grad_norm)

        self.optimizer.step()

        return info