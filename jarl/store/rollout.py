from dataclasses import dataclass

import torch as th

from jarl.data.batch import TensorBatch
from jarl.store.base import TensorStorage


@dataclass(frozen=True)
class Rollout:
    steps: TensorBatch

    def with_steps(self, steps: TensorBatch) -> "Rollout":
        return Rollout(steps)


class RolloutBuffer(TensorStorage):
    def __init__(
        self,
        horizon: int,
        num_envs: int,
        device: str | th.device,
        copy_on_finish: bool = True,
    ) -> None:
        super().__init__(horizon, num_envs, device)
        self.horizon = horizon
        self.copy_on_finish = copy_on_finish
        self.position = 0

    @property
    def full(self) -> bool:
        return self.position == self.horizon

    def append(self, transition: dict[str, object]) -> None:
        if self.full:
            raise RuntimeError("rollout is full")

        self._write(self.position, transition)
        self.position += 1

    def finish(self) -> Rollout:
        if self.position == 0 or self._storage is None:
            raise RuntimeError("cannot finish an empty rollout")

        steps = {
            key: (
                value[: self.position].clone()
                if self.copy_on_finish
                else value[: self.position]
            )
            for key, value in self._storage.items()
        }
        return Rollout(TensorBatch(steps))

    def clear(self) -> None:
        self.position = 0
