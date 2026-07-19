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
    ) -> None:
        super().__init__(horizon, num_envs, device)
        self.horizon = horizon
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

        return Rollout(
            TensorBatch(
                {
                    key: value[: self.position].clone()
                    for key, value in self._storage.items()
                }
            )
        )

    def clear(self) -> None:
        self.position = 0
