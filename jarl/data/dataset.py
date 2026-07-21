from dataclasses import dataclass
from typing import Any

import torch as th

from jarl.data.batch import TensorBatch


@dataclass(frozen=True)
class TensorDataset:
    data: TensorBatch

    def __post_init__(self) -> None:
        if not isinstance(self.data, TensorBatch):
            raise TypeError("data must be a TensorBatch")
        if not len(self.data):
            raise ValueError("a tensor dataset cannot be empty")

        devices = {value.device for value in self.data.values()}
        if len(devices) != 1:
            raise ValueError("tensor dataset fields must share a device")

    @property
    def device(self) -> th.device:
        return self.data.device

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, indices: Any) -> TensorBatch:
        if isinstance(indices, int):
            if indices < 0:
                indices += len(self)
            if not 0 <= indices < len(self):
                raise IndexError("tensor dataset index out of range")
            indices = slice(indices, indices + 1)
        elif isinstance(indices, th.Tensor) and indices.ndim == 0:
            indices = indices.reshape(1)
        return self.data[indices]

    def sample(
        self,
        count:     int,
        generator: th.Generator | None = None,
    ) -> TensorBatch:
        if count < 1:
            raise ValueError("sample count must be positive")
        indices = th.randint(
            len(self),
            (count,),
            device=self.device,
            generator=generator,
        )
        return self[indices]

    def to(
        self,
        device:       str | th.device,
        non_blocking: bool = False,
    ) -> "TensorDataset":
        return TensorDataset(self.data.to(device, non_blocking=non_blocking))
