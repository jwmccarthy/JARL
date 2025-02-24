import numpy as np
from numpy import ndarray

import torch as th
from torch import Tensor

from dataclasses import dataclass, field

from jarl.data.types import Device, torch_to_numpy


@dataclass
class TensorSpec:
    """Tensor spec w/ shape, dtype, & numpy conversion"""

    shape:  tuple
    dtype:  th.dtype
    stype:  np.dtype = field(init=False)
    device: Device = field(default="cpu", kw_only=True)

    def __post_init__(self) -> None:
        self.stype = torch_to_numpy[self.dtype]

    def __call__(self, x: ndarray) -> Tensor:
        if not isinstance(x, ndarray):
            x = np.array(x)
        assert self.shape == x.shape
        assert self.stype == x.dtype
        return th.from_numpy(x).to(self.device)