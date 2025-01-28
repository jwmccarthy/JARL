from torch import Tensor

from abc import ABC, abstractmethod
from multimethod import multimethod
from typing import Self, Tuple, Iterable

from jarl.data.types import (
    Index, Device, 
    NestedIterable,
    NestedTensorDict
)


class MultiIterable(ABC):
    """
    Base class for nested indexable storage of multiple iterables
    """

    def __init__(
        self, 
        data: NestedIterable
    ) -> None:
        for key, val in data.items():
            data[key] = self._parse(val)
        self._data = data

    @abstractmethod
    def _parse(self, val: Iterable | NestedIterable) -> Iterable | Self:
        ...

    @multimethod
    def __getitem__(self, idx: str) -> Iterable | Self:
        return self._data[idx]
    
    @multimethod
    def __getitem__(self, idx: Index) -> Self:
        data = {k: v[idx] for k, v in self.items()}
        return self.__class__(data)
    
    def __len__(self) -> int:
        return len(next(iter(self._data.values())))
    
    def __setitem__(self, idx: Index, val: NestedIterable) -> None:
        for k, v in val.items():
            self._data[k][idx] = self._parse(v)

    def __getattr__(self, key: str) -> Iterable | Self:
        return self._data[key]  # __getattribute__ for missing keys

    def __contains__(self, key: str) -> bool:
        return key in self._data
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data})"

    def items(self) -> Iterable[Tuple[str, Iterable | Self]]:
        return self._data.items()
    
    def keys(self) -> Iterable[str]:
        return self._data.keys()
    
    def set(self, **data: NestedIterable) -> Self:
        for key, val in data.items():
            self._data[key] = self._parse(val)
        return self


class MultiTensor(MultiIterable):
    """
    Indexable nested storage for tensors with device management
    """

    def __init__(
        self, 
        tensors: NestedTensorDict,
        device: Device = "cpu"
    ) -> None:
        self._device = device
        super().__init__(tensors)

    def _parse(self, val: Tensor | NestedTensorDict) -> Tensor | Self:
        if isinstance(val, dict):
            return MultiTensor(val, self.device)
        return val.to(self.device)

    @property
    def device(self) -> Device:
        return self._device

    def to(self, device: Device) -> Self:
        return MultiTensor(self._data, device)