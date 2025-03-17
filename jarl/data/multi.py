import numpy as np
import torch as th

from abc import ABC

from collections import defaultdict
from multimethod import multimethod

from numpy.typing import NDArray
from typing import Self, Mapping, Tuple, List, Iterable, Any

from jarl.data.dict import DotDict
from jarl.data.utils import common_shape
from jarl.data.types import Device, Index, NumpyIndex, TorchIndex


# TODO: expand base class
class MultiIterable(ABC, dict):

    def __init__(self, *args, **kwargs: Mapping[str, Iterable]) -> None:
        super().__init__(*args, **kwargs)
    
    def __getattr__(self, key: str) -> Any:
        return self[key] if key in self else super().__getattribute__(key)


class MultiArray(MultiIterable):

    def __init__(
        self, 
        *args: Tuple[Any], 
        **kwargs: Mapping[str, NDArray]
    ) -> None:
        super().__init__(*args, **kwargs)
        self._common = common_shape(self.values())
        assert not self.keys() or self._common, (
            "Arrays require at least one shared dimension")
    
    @property
    def shape(self) -> Tuple[int]:
        return self._common
    
    def __len__(self) -> int:
        return self.shape[0]

    @multimethod
    def __setattr__(self, key: str, val: Any) -> None:
        super().__setattr__(key, val)

    @__setattr__.register
    def __setattr__(self, key: str, val: NDArray) -> None:
        super().__setitem__(key, val)
  
    @multimethod
    def __getitem__(self, idx: Any) -> Any:
        return super().__getitem__(idx)
    
    @__getitem__.register
    def _(self, idx: int) -> DotDict:
        return DotDict(**{k: v[idx] for k, v in self.items()})
    
    @__getitem__.register
    def _(self, idx: NumpyIndex | th.Tensor) -> Self:
        return self.__class__(**{k: v[idx] for k, v in self.items()})

    @multimethod
    def __setitem__(self, key: str, val: Any) -> None:
        super().__setitem__(key, val)

    @__setattr__.register
    def _(self, key: str, val: NDArray) -> None:
        assert val.shape[:len(self.shape)] == self.shape
        self[key] = val

    @__setitem__.register
    def _(self, idx: NumpyIndex, data: Mapping[str, NDArray]) -> None:
        for key, val in data.items(): self[key][idx] = val

    def flatten(self, start: int, end: int) -> Self:
        data = {}
        for key, val in self.items():
            shape = val.shape
            new_shape = shape[:start] + (-1,) + shape[end + 1:]
            data[key] = val.reshape(new_shape)
        return MultiArray(**data)



class MultiTensor(MultiIterable):

    def __init__(
        self, 
        *args: Tuple[Any],
        device: Device = "cpu", 
        **kwargs: Mapping[str, th.Tensor]
    ) -> None:
        super().__init__(*args, **kwargs)
        self._device = device
        self._common = common_shape(self.values())
        assert not self.keys() or self._common, (
            "Arrays require at least one shared dimension")
        for key, val in self.items():
            self[key] = val.to(self._device)
    
    @classmethod
    def from_numpy(
        cls, 
        data: Mapping[str, NDArray], 
        device: Device
    ) -> Self:
        data = {k: th.tensor(v) for k, v in data.items()}
        return cls(**data, device=device)

    @property
    def shape(self) -> Tuple[int]:
        return self._common
    
    @property
    def device(self) -> Device:
        return self._device
    
    def __len__(self) -> int:
        return self.shape[0]

    @multimethod
    def __setattr__(self, key: str, val: Any) -> None:
        super().__setattr__(key, val)

    @__setattr__.register
    def __setattr__(self, key: str, val: th.Tensor) -> None:
        super().__setitem__(key, val)
  
    @multimethod
    def __getitem__(self, idx: Any) -> Any:
        return super().__getitem__(idx)
    
    @__getitem__.register
    def _(self, idx: int) -> DotDict:
        return DotDict(**{k: v[idx] for k, v in self.items()})
    
    @__getitem__.register
    def _(self, idx: TorchIndex | th.Tensor) -> Self:
        return self.__class__(**{k: v[idx] for k, v in self.items()})

    @multimethod
    def __setitem__(self, key: str, val: Any) -> None:
        super().__setitem__(key, val)

    @__setattr__.register
    def _(self, key: str, val: th.Tensor) -> None:
        assert val.shape[:len(self.shape)] == self.shape
        self[key] = val

    @__setitem__.register
    def _(self, idx: NumpyIndex, data: Mapping[str, th.Tensor]) -> None:
        for key, val in data.items(): self[key][idx] = val

    def to(self, device: Device) -> Self:
        self._device = device
        for key, val in self.items():
            self[key] = val.to(device)
        return self
    
    def append(self, data: Mapping[str, th.Tensor]) -> Self:
        assert self.keys() == data.keys(), "Keys must match"

        # convert to MultiTensor if necessary
        if not isinstance(data, MultiTensor):
            data = MultiTensor(**data, device=self.device)

        # adjust common shape for added data
        self._common = (len(self) + len(data),) + self.shape[1:]

        for key, val in data.items():
            assert self[key].shape[1:] == val.shape[1:], (
                f"Shape mismatch for key '{key}'")
            self[key] = th.cat([self[key], val], dim=0)

        return self
    
    def flatten(self, start: int, end: int) -> th.Tensor:
        data = {k: v.flatten(start, end) for k, v in self.items()}
        return MultiTensor(**data, device=self.device)
    

class MultiList(defaultdict):

    def __init__(self, **kwargs: Mapping[str, List]) -> None:
        super().__init__(list)
        for key, val in kwargs.items():
            self[key] = val

    def __getattr__(self, key: str) -> List:
        return self[key] if key in self else super().__getattribute__(key)

    @multimethod
    def __setattr__(self, key: str, val: Any) -> None:
        super().__setattr__(key, val)

    @__setattr__.register
    def __setattr__(self, key: str, val: List) -> None:
        super().__setitem__(key, val)

    @multimethod
    def __getitem__(self, idx: str) -> List:
        return super().__getitem__(idx)

    @multimethod
    def __getitem__(self, idx: int) -> DotDict:
        return DotDict(**{k: v[idx] for k, v in self.items()})
    
    @multimethod
    def __getitem__(self, idx: slice) -> Self:
        return MultiList(**{k: v[idx] for k, v in self.items()})
    
    @multimethod
    def __setitem__(self, idx: str, data: List | "MultiList") -> None:
        super().__setitem__(idx, data)
    
    @multimethod
    def __setitem__(self, idx: Index, data: Mapping[str, Any]) -> Self:
        for key, val in data.items():
            self[key][idx] = val

    def append(self, data: Mapping[str, Any]) -> Self:
        for key, val in data.items():
            self[key].append(val)
        return self
    
    def extend(self, data: Mapping[str, List[Any]]) -> Self:
        for key, val in data.items():
            self[key].extend(val)
        return self