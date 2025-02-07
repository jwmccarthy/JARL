import torch as th

from typing import Iterable


def common_shape(tensors: Iterable[th.Tensor]) -> th.Size:
    common = ()
    shapes = [val.shape for val in tensors]
    for group in zip(*shapes):
        if len(set(group)) > 1:
            break
        common += (group[0],)
    return th.Size(common)