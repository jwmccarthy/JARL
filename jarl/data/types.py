import numpy as np
import torch as th
from torch import Tensor

from typing import (
    Iterable,
    Union, 
    Dict, 
    Callable
)


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
Device = str | th.device

# nested dicts
NestedIterable = Dict[str, Union[Iterable, "NestedIterable"]]
NestedTensorDict = Dict[str, Union[Tensor, "NestedTensorDict"]]