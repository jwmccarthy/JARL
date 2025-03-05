import torch as th

from collections import defaultdict
from multimethod import multimethod
from typing import (
    Dict, Self, Any, Iterable, Mapping, Tuple, List
)

from jarl.data.dict import DotDict
from jarl.data.utils import common_shape
from jarl.data.types import Device, Index


class MultiTensor(dict):

    def __init__(
        self, 
        *args,
        device: Device = "cpu",
        **kwargs: Dict[str, th.Tensor] 
    ) -> None:
        super().__init__(*args, **kwargs)
        self._device = device
        self._common = common_shape(self.values())

        # allow empty initialization
        assert not self.keys() or self._common, (
            "Tensors require at least one shared dimensinon")
        
        # move tensors to specified device
        for key, val in self.items():
            self[key] = val.to(self.device)

    @property
    def shape(self) -> tuple[int]:
        return self._common
    
    @property
    def device(self) -> Device:
        return self._device
    
    def __len__(self) -> int:
        return self.shape[0]
    
    def __getattr__(self, key: str) -> th.Tensor:
        if key in self:
            return self[key]
        return super().__getattribute__(key)

    @multimethod
    def __setattr__(self, key: str, val: th.Tensor) -> None:
        super().__setitem__(key, val)

    @multimethod
    def __setattr__(self, key: str, val: Any) -> None:
        super().__setattr__(key, val)

    @multimethod
    def __getitem__(self, idx: str) -> th.Tensor:
        return super().__getitem__(idx)
    
    @multimethod
    def __getitem__(self, keys: List[str]) -> Self:
        data = {k: self[k] for k in keys}
        return MultiTensor(**data)

    @multimethod
    def __getitem__(self, idx: int) -> Self:
        data = {k: v[idx] for k, v in self.items()}
        return DotDict(**data)
    
    @multimethod
    def __getitem__(self, idx: tuple | Index) -> Self:
        data = {k: v[idx] for k, v in self.items()}
        return MultiTensor(**data, device=self.device)
    
    @multimethod
    def __setitem__(self, idx: str, data: th.Tensor) -> None:
        assert data.shape[:len(self.shape)] == self.shape
        super().__setitem__(idx, data)
    
    @multimethod
    def __setitem__(self, idx: Index, data: Mapping[str, th.Tensor]) -> Self:
        for key, val in data.items():
            self[key][idx] = val

    def __new__(cls, **kwargs):
        instance = super().__new__(cls)
        instance._common = ()  
        instance._device = kwargs.get("device", "cpu")
        return instance

    def __setstate__(self, state: dict) -> None:
        self.update(**state["data"])
        self._device = state["device"]
        self._common = common_shape(self.values())

    def __getstate__(self) -> dict:
        return dict(data=dict(self), device=self.device)

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
    
    def split(self, sections: int | Iterable[int]) -> Tuple[Self]:
        cls = MultiTensor if self.shape[1:] else DotDict
        dols = {k: v.split(sections) for k, v in self.items()}
        lods = [dict(zip(dols.keys(), vals)) for vals in zip(*dols.values())]
        return tuple(cls(**dol) for dol in lods)
    
    def sample(self, size: int) -> Self:
        assert size <= len(self), "Sample size exceeds buffer size"
        return self[th.randperm(len(self))[:size]]
    
    def flatten(self, start_dim: int, end_dim: int) -> th.Tensor:
        data = {k: v.flatten(start_dim, end_dim) for k, v in self.items()}
        return MultiTensor(**data, device=self.device)
    

class MultiList(defaultdict):

    def __init__(self, **kwargs: Dict[str, List]) -> None:
        super().__init__(list)
        for key, val in kwargs.items():
            self[key] = val

    def __getattr__(self, key: str) -> List:
        if key in self:
            return self[key]
        return super().__getattribute__(key)

    @multimethod
    def __setattr__(self, key: str, val: List) -> None:
        super().__setitem__(key, val)

    @multimethod
    def __setattr__(self, key: str, val: Any) -> None:
        super().__setattr__(key, val)

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