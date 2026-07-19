from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch as th


@dataclass(frozen=True)
class TensorBatch(Mapping[str, th.Tensor]):
    data: dict[str, th.Tensor]

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("a tensor batch cannot be empty")
        if not self.shape:
            raise ValueError("batch tensors require a shared leading dimension")

    @property
    def shape(self) -> tuple[int, ...]:
        shapes = [tuple(value.shape) for value in self.data.values()]
        common = []
        for dimensions in zip(*shapes):
            if len(set(dimensions)) != 1:
                break
            common.append(dimensions[0])
        return tuple(common)

    @property
    def device(self) -> th.device:
        return next(iter(self.data.values())).device

    def __getitem__(self, key: str | Any):
        if isinstance(key, str):
            return self.data[key]
        return TensorBatch({name: value[key] for name, value in self.data.items()})

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return self.shape[0]

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def select(self, *keys: str) -> "TensorBatch":
        return TensorBatch({key: self.data[key] for key in keys})

    def with_fields(self, **fields: th.Tensor) -> "TensorBatch":
        duplicate = self.data.keys() & fields.keys()
        if duplicate:
            raise KeyError(f"fields already exist: {sorted(duplicate)}")
        for key, value in fields.items():
            prefix = min(value.ndim, len(self.shape))
            if prefix == 0 or tuple(value.shape[:prefix]) != self.shape[:prefix]:
                raise ValueError(
                    f"field {key!r} does not share a leading batch shape"
                )
        return TensorBatch(self.data | fields)

    def replace_fields(self, **fields: th.Tensor) -> "TensorBatch":
        missing = fields.keys() - self.data.keys()
        if missing:
            raise KeyError(f"fields do not exist: {sorted(missing)}")
        for key, value in fields.items():
            if value.shape != self.data[key].shape:
                raise ValueError(f"replacement field {key!r} changed shape")
        return TensorBatch(self.data | fields)

    def flatten(self, start: int, end: int) -> "TensorBatch":
        return TensorBatch(
            {key: value.flatten(start, end) for key, value in self.data.items()}
        )

    def to(self, device: str | th.device, non_blocking: bool = False) -> "TensorBatch":
        return TensorBatch(
            {
                key: value.to(device, non_blocking=non_blocking)
                for key, value in self.data.items()
            }
        )
