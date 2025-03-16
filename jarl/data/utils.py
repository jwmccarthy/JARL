import numpy as np
import torch as th

from typing import Iterable


def common_shape(arrays: Iterable[th.Tensor | np.ndarray]) -> th.Size:
    common = ()
    shapes = [val.shape for val in arrays]
    for group in zip(*shapes):
        if len(set(group)) > 1:
            break
        common += (group[0],)
    return common