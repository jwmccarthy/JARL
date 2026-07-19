from dataclasses import dataclass

import torch as th

from jarl.data.batch import TensorBatch


@dataclass(frozen=True)
class Rollout:
    steps: TensorBatch

    def with_steps(self, steps: TensorBatch) -> "Rollout":
        return Rollout(steps)


class RolloutBuffer:
    def __init__(
        self,
        horizon: int,
        num_envs: int,
        device: str | th.device,
    ) -> None:
        if horizon < 1 or num_envs < 1:
            raise ValueError("rollout dimensions must be positive")
        self.horizon = horizon
        self.num_envs = num_envs
        self.device = th.device(device)
        self.position = 0
        self._storage: dict[str, th.Tensor] | None = None

    @property
    def full(self) -> bool:
        return self.position == self.horizon

    def _initialize(self, record: dict[str, object]) -> None:
        storage = {}
        for key, value in record.items():
            tensor = th.as_tensor(value, device=self.device)
            if len(tensor) != self.num_envs:
                raise ValueError(
                    f"field {key!r} has {len(tensor)} environments, "
                    f"expected {self.num_envs}"
                )
            storage[key] = th.empty(
                (self.horizon, *tensor.shape),
                dtype=tensor.dtype,
                device=self.device,
            )
        self._storage = storage

    def append(self, record: dict[str, object]) -> None:
        if self.full:
            raise RuntimeError("rollout is full")
        if self._storage is None:
            self._initialize(record)
        if record.keys() != self._storage.keys():
            raise KeyError("rollout record fields changed after initialization")

        for key, value in record.items():
            tensor = th.as_tensor(value, device=self.device)
            expected = self._storage[key].shape[1:]
            if tensor.shape != expected:
                raise ValueError(
                    f"field {key!r} has shape {tuple(tensor.shape)}, "
                    f"expected {tuple(expected)}"
                )
            self._storage[key][self.position].copy_(tensor)
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
