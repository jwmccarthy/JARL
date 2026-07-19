from typing import Protocol

import torch as th

from jarl.data.records import ActionDecision, Evaluation


class Behavior(Protocol):
    device: str | th.device

    def initial_state(self, batch_size: int) -> th.Tensor | None:
        ...

    def act(
        self,
        obs: th.Tensor,
        state: th.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> ActionDecision:
        ...


class TrainableBehavior(Behavior, Protocol):
    def evaluate_actions(
        self,
        obs: th.Tensor,
        action: th.Tensor,
        state: th.Tensor | None = None,
        *,
        reset: th.Tensor | None = None,
    ) -> Evaluation:
        ...
