import numpy as np
import torch as th
from torch import nn
from collections.abc import Callable
from typing import Iterable


class LayerInit:
    """A composable operation for initializing a torch layer."""

    def __init__(self, initializers: Iterable[Callable[[nn.Module], None]]) -> None:
        self.initializers = tuple(initializers)

    def __call__(self, layer: nn.Module) -> nn.Module:
        for initializer in self.initializers:
            initializer(layer)
        return layer

def orthogonal_init(
    std:        float = np.sqrt(2),
    bias_const: float = 0.0,
) -> LayerInit:
    def init_weight(layer: nn.Module) -> None:
        th.nn.init.orthogonal_(layer.weight, std)

    def init_bias(layer: nn.Module) -> None:
        if layer.bias is not None:
            th.nn.init.constant_(layer.bias, bias_const)

    return LayerInit((init_weight, init_bias))
