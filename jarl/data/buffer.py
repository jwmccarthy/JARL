import torch as th
from torch import Tensor

from typing import Self
from abc import ABC, abstractmethod

from jarl.data.multi import MultiTensor
from jarl.data.types import Device, NestedTensorDict


class Buffer(ABC):

    @abstractmethod
    def store(self, data: NestedTensorDict) -> None:
        ...

    @abstractmethod
    def serve(self) -> MultiTensor:
        ...


class LazyBuffer(Buffer):
    """Replay buffer for storing transitions"""

    def __init__(
        self, 
        size: int,
        device: Device = "cpu"
    ) -> None:
        super().__init__()
        self._idx = 0
        self.size = size
        self.full = False
        self.data = None
        self.device = device

    def _nested_init(
        self, 
        data: Tensor | NestedTensorDict
    ) -> Tensor | MultiTensor:
        # initialize tensors w/ shape & dtype
        if isinstance(data, Tensor):
            dtype = data.dtype
            shape = (self.size, *data.shape)
            return th.empty(shape, dtype=dtype, device=self.device)
        
        # recursively initialize nested tensors
        tensors = {k: self._nested_init(v) for k, v in data.items()}
        return MultiTensor(tensors, self.device)

    def _lazy_init(self, data: NestedTensorDict) -> None:
        self.data = self._nested_init(data)

    def __len__(self) -> int:
        return self.size if self.full else self._idx
    
    def to(self, device: Device) -> Self:
        self.device = device
        if self.data is not None:
            self.data.to(device)
        return self
    
    def reset(self) -> None:
        self._idx = 0
        self.full = False

    def clear(self) -> None:
        self.reset()
        self.data = None

    def store(self, data: NestedTensorDict) -> None:
        # lazy-initialize tensor storage
        if self._idx == 0 and not self.full:
            self._lazy_init(data)

        # store data circularly
        self.data[self._idx] = data
        self._idx = (self._idx + 1) % self.size

        # full circular pass
        if self._idx == 0:
            self.full = True

    def serve(self) -> MultiTensor:
        if self.full:
            return self.data[:]  # slice idx prevents overwriting
        return self.data[:self._idx]