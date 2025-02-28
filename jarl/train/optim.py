import torch.optim as opt
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import _LRScheduler

from itertools import chain


class Optimizer:

    def __init__(
        self, 
        optimizer: opt.Optimizer, 
        max_grad_norm: float = 0.5, 
        **op_kwargs
    ) -> None:
        self.optimizer = optimizer
        self.op_kwargs = op_kwargs
        self.max_grad_norm = max_grad_norm

    def build(self, modules):
        self.params = chain(*[m.parameters() for m in modules])
        self.optimizer = self.optimizer(self.params, **self.op_kwargs)
        return self
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

        # clip gradients
        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)

        self.optimizer.step()


class Scheduler:

    def __init__(
        self, 
        scheduler: _LRScheduler, 
        **lr_kwargs
    ) -> None:
        self.optimizer = None
        self.scheduler = scheduler
        self.lr_kwargs = lr_kwargs

    def build(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer.optimizer
        return self
    
    def start(self, steps: int) -> None:
        self.scheduler = self.scheduler(
            self.optimizer, total_iters=steps, **self.lr_kwargs
        )
    
    def step(self):
        self.scheduler.step()
