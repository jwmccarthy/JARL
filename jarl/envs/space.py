import numpy as np

import torch as th
from torch import Tensor

from gymnasium.spaces import (
    Space,
    Box,
    Discrete,
    MultiDiscrete,
    MultiBinary
)

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from jarl.data.spec import TensorSpec
from jarl.data.types import Device, numpy_to_torch


@dataclass
class TensorSpace(TensorSpec, ABC):
    """Gymnasium space as tensor spec"""

    space: Space
    shape: tuple = field(init=False)
    dtype: th.dtype = field(init=False)
    stype: np.dtype = field(init=False)

    def __post_init__(self) -> None:
        self.shape = self.space.shape
        self.stype = self.space.dtype.type
        self.dtype = numpy_to_torch[self.stype]

    def contains(self, x: Tensor) -> bool:
        np_x = x.detach().cpu().numpy()
        return self.space.contains(np_x)
    
    def sample(self) -> Tensor:
        x = self.space.sample()
        return self(x)
    
    @property
    @abstractmethod
    def flat_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def numel(self) -> int:
        ...


@dataclass
class BoxSpace(TensorSpace):
    """Tensor spec for Box gym space"""

    _low:  Tensor = field(init=False)
    _high: Tensor = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._low = th.as_tensor(self.space.low, dtype=self.dtype)
        self._high = th.as_tensor(self.space.high, dtype=self.dtype)

    @property
    def low(self) -> Tensor:
        return self._low.to(self.device)
    
    @property
    def high(self) -> Tensor:
        return self._high.to(self.device)
    
    @property
    def flat_dim(self) -> int:
        return np.prod(self.shape)
    
    @property
    def numel(self) -> int:
        return self.shape[0]
    

@dataclass
class DiscreteSpace(TensorSpace):
    """Tensor spec for Discrete gym space"""

    n: int = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.n = self.space.n
    
    @property
    def flat_dim(self) -> int:
        return self.n
    
    @property
    def numel(self) -> int:
        return 1
    

@dataclass
class MultiDiscreteSpace(TensorSpace):
    """Tensor spec for MultiDiscrete gym space"""

    _nvec: Tensor = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._nvec = th.from_numpy(self.space.nvec)

    @property
    def nvec(self) -> Tensor:
        return self._nvec.to(self.device)
    
    @property
    def flat_dim(self) -> int:
        return np.sum(self.nvec)
    
    @property
    def numel(self) -> int:
        return len(self.nvec)
    

@dataclass
class MultiBinarySpace(TensorSpace):
    """Tensor spec for MultiBinary gym space"""

    _n: int | tuple | np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._n = th.as_tensor(self.space.n, dtype=th.int32)
    
    @property
    def n(self) -> Tensor:
        return self._n.to(self.device)

    @property
    def flat_dim(self) -> int:
        return np.prod(self._n)
    
    @property
    def numel(self) -> int:
        return len(self._n)
    

@dataclass
class ConcatSpace(TensorSpec):

    space: TensorSpace
    count: int
    shape: tuple = field(init=False)
    dtype: th.dtype = field(init=False)
    stype: np.dtype = field(init=False)

    def __post_init__(self) -> None:
        self.shape = (self.count * self.space.shape[0],) \
                   + (*self.space.shape[1:],)
        self.stype = self.space.stype
        self.dtype = self.space.dtype

    def __getattr__(self, key: str) -> Tensor:
        if key in self.__dict__:
            super().__getattribute__(key)
        return self.space.__getattribute__(key)

    def contains(self, x: Tensor) -> bool:
        return all(self.space.contains(y) for y in x)
    
    def sample(self) -> Tensor:
        return th.cat([self.space.sample() for _ in range(self.count)], 0)

    @property
    def flat_dim(self) -> int:
        return self.space.flat_dim * self.count
    
    @property
    def numel(self) -> int:
        return self.space.numel * self.count
    

@dataclass
class StackedSpace(TensorSpec):
    space: TensorSpace
    count: int
    shape: tuple = field(init=False)
    dtype: th.dtype = field(init=False)
    stype: np.dtype = field(init=False)

    def __post_init__(self) -> None:
        self.shape = (self.count,) + self.space.shape
        self.stype = self.space.stype
        self.dtype = self.space.dtype

    def __getattr__(self, key: str) -> Tensor:
        if key in self.__dict__:
            super().__getattribute__(key)
        return self.space.__getattribute__(key)

    def contains(self, x: Tensor) -> bool:
        return all(self.space.contains(y) for y in x)
    
    def sample(self) -> Tensor:
        return th.stack([self.space.sample() for _ in range(self.count)])

    @property
    def flat_dim(self) -> int:
        return self.space.flat_dim * self.count
    
    @property
    def numel(self) -> int:
        return self.space.numel * self.count


def torch_space(space: Space, device: Device = "cpu") -> TensorSpace:
    """Convert gym space to tensor space"""
    if isinstance(space, Box):
        return BoxSpace(space, device=device)
    if isinstance(space, Discrete):
        return DiscreteSpace(space, device=device)
    if isinstance(space, MultiDiscrete):
        return MultiDiscreteSpace(space, device=device)
    if isinstance(space, MultiBinary):
        return MultiBinarySpace(space, device=device)
    raise TypeError(f"Unsupported space type: {type(space)}")