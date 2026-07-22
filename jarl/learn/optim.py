from collections.abc import Iterable

import torch as th
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_


def unique_parameters(modules: nn.Module | Iterable[nn.Module]):
    if isinstance(modules, nn.Module):
        modules = (modules,)
    seen = set()
    parameters = []

    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    return parameters


class OptimizerStep:
    def __init__(
        self,
        modules: nn.Module | Iterable[nn.Module],
        optimizer: th.optim.Optimizer,
        max_grad_norm: float | None = None,
        scheduler=None,
    ) -> None:
        self.parameters = unique_parameters(modules)
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler

    def __call__(self, loss: th.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm is not None:
            clip_grad_norm_(self.parameters, self.max_grad_norm)
        self.optimizer.step()

    def advance_scheduler(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()


class IndependentOptimizerSteps:
    """Apply one backward pass with independent clipping and optimizers."""

    def __init__(self, *steps: OptimizerStep) -> None:
        if not steps:
            raise ValueError("at least one optimizer step is required")
        parameter_ids = [
            id(parameter) for step in steps for parameter in step.parameters
        ]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("independent optimizer steps cannot share parameters")
        self.steps = steps

    def __call__(self, loss: th.Tensor) -> None:
        for step in self.steps:
            step.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for step in self.steps:
            if step.max_grad_norm is not None:
                clip_grad_norm_(step.parameters, step.max_grad_norm)
            step.optimizer.step()

    def advance_scheduler(self) -> None:
        for step in self.steps:
            step.advance_scheduler()
