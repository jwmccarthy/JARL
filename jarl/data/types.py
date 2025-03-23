import numpy as np
import torch as th
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from typing import (
    Iterable,
    Dict, 
    Tuple,
    Any, 
    Generator,
    Callable
)
from numpy.typing import NDArray

from jarl.data.dict import DotDict


numpy_to_torch = {
    np.uint8      : th.uint8,
    np.int8       : th.int8,
    np.int16      : th.int16,
    np.int32      : th.int32,
    np.int64      : th.int64,
    np.float16    : th.float16,
    np.float32    : th.float32,
    np.float64    : th.float64,
    np.complex64  : th.complex64,
    np.complex128 : th.complex128
}

torch_to_numpy = {
    v: k for k, v in numpy_to_torch.items()
}

# composite types
Index = int | slice | Iterable[int]
NumpyIndex = Index | NDArray[int | bool]
TorchIndex = Index | th.Tensor

Device = str | th.device

# computations
LossInfo = Tuple[th.Tensor, Dict[str, Any]]
SampleOutput = Generator[Iterable, None, None] | Tuple[Iterable, ...]
EnvStep = Tuple[Tuple[th.Tensor, ...], Dict[str, Any]]
EnvOutput = Tuple[DotDict[str, th.Tensor], th.Tensor]
SchedulerFunc = Callable[[Optimizer, int], LRScheduler]