import torch as th

from multimethod import multimethod
from typing import Dict, Self, Any, Mapping

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
        assert not self.keys() or self._common

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
    def __getitem__(self, idx: int) -> Self:
        data = {k: v[idx] for k, v in self.items()}
        return DotDict(**data)
    
    @multimethod
    def __getitem__(self, idx: Index) -> Self:
        data = {k: v[idx] for k, v in self.items()}
        return MultiTensor(**data, device=self.device)
    
    @multimethod
    def __setitem__(self, idx: str, data: th.Tensor) -> None:
        # assert data.shape[:len(self.shape)] == self.shape
        super().__setitem__(idx, data)
    
    @multimethod
    def __setitem__(self, idx: Index, data: Mapping[str, th.Tensor]) -> Self:
        for key, val in data.items():
            self[key][idx] = val

    def to(self, device: Device) -> Self:
        self._device = device
        for key, val in self.items():
            self[key] = val.to(device)
        return self