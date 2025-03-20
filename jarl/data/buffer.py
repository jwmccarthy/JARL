import numpy as np
import torch as th

from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import Dict, TypeVar, Generic

from jarl.data.types import Device
from jarl.data.multi import MultiIterable, MultiArray, MultiTensor


T = TypeVar("T", NDArray, th.Tensor)


class Buffer(ABC, Generic[T]):

    _data: MultiIterable

    def __init__(self, size: int) -> None:
        self._idx = 0
        self._size = size
        self._full = False

    @abstractmethod
    def store(self, data: Dict[str, T]) -> None:
        ...

    @abstractmethod
    def serve(self) -> MultiIterable:
        ...


class LazyBuffer(Buffer):

    _data: MultiArray

    def __init__(self, size: int) -> None:
        super().__init__(size)
        self._init = False

    def _lazy_init(self, batch: Dict[str, th.Tensor]) -> None:
        data, self._init = {}, True
        for key, val in batch.items():
            shape = (self._size, *val.shape)
            data[key] = np.empty(shape, dtype=val.dtype)
        self._data = MultiArray(**data)

    def __len__(self) -> int:
        return self.size if self._full else self._idx

    def store(self, data: Dict[str, th.Tensor]) -> None:
        # lazy-initialize tensor storage
        if not self._init:
            self._lazy_init(data)

        # store data circularly
        self._data[self._idx] = data
        self._idx = (self._idx + 1) % self._size

        # full circular pass
        if self._idx == 0:
            self._full = True

    def serve(self) -> MultiArray:
        out = self._data[:] if self._full else self._data[:self._idx]
        return MultiTensor.from_numpy(out, device="cuda")