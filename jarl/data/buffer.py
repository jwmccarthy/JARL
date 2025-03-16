import torch as th
from torch import Tensor

from typing import Self, Dict
from abc import ABC, abstractmethod

from jarl.data.types import Device
from jarl.data.multi import MultiTensor


class Buffer(ABC):

    @abstractmethod
    def store(self, data: Dict[str, th.Tensor]) -> None:
        ...

    @abstractmethod
    def serve(self) -> MultiTensor:
        ...


class LazyBuffer(Buffer):
    """Replay buffer for storing transitions"""

    _data: MultiTensor

    def __init__(
        self, 
        size: int,
        device: Device = "cpu"
    ) -> None:
        super().__init__()
        self._idx = 0
        self._size = size      # max buffer size before circling
        self._init = False     # lazy initialization flag
        self._full = False     # full buffer flag
        self._device = device  # device to store data
    
    @property
    def size(self) -> int:
        return self._size

    @property
    def init(self) -> bool:
        return self._init

    @property
    def full(self) -> bool:
        return self._full
    
    @property
    def device(self) -> Device:
        return self._device

    def _lazy_init(self, batch: Dict[str, th.Tensor]) -> None:
        data, self._init = {}, True
        for key, val in batch.items():
            shape = (self.size, *val.shape)
            data[key] = th.empty(shape, dtype=val.dtype)
        self._data = MultiTensor(**data, device=self.device)

    def __len__(self) -> int:
        return self.size if self.full else self._idx
    
    def to(self, device: Device) -> Self:
        self._device = device
        if self.init:
            self._data.to(device)
        return self

    def reset(self) -> Self:
        for key, val in self._data.items():
            self._data[key] = val.zero_()
        return self

    def store(self, data: Dict[str, th.Tensor]) -> None:
        # lazy-initialize tensor storage
        if not self.init:
            self._lazy_init(data)

        # store data circularly
        self._data[self._idx] = data
        self._idx = (self._idx + 1) % self.size

        # full circular pass
        if self._idx == 0:
            self._full = True

    def serve(self) -> MultiTensor:
        if self.full:
            return self._data[:]
        return self._data[:self._idx]