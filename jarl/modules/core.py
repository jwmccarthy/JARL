import torch.nn as nn
from torch import Tensor

from typing import List, Self

from jarl.modules.utils import init_layer


class MLP(nn.Module):

    model: nn.Module

    def __init__(
        self, 
        dims: List[int] = [64, 64],
        func: nn.Module = nn.ReLU,
        init_func=init_layer
    ) -> None:
        super().__init__()
        self.dims = dims            # hidden layer dims
        self.func = func            # activation function
        self.init_func = init_func  # weight init function

    def build(self, in_dim: int, out_dim: int) -> Self:
        self.model = nn.Sequential()

        # dims define hidden layers
        for next_dim in self.dims:
            layer = self.init_func(nn.Linear(in_dim, next_dim))
            self.model.extend([layer, self.func()])
            in_dim = next_dim

        # output linear layer
        self.model.append(nn.Linear(in_dim, out_dim))

        return self

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    

class CNN(nn.Module):

    model: nn.Module

    def __init__(
        self,
        func: nn.Module = nn.ReLU,
        dims: List[int] = [32, 64],
        kernel: List[int] = [8, 4],
        stride: List[int] = [4, 2],
        init_func=init_layer
    ) -> None:
        super().__init__()
        self.dims = dims
        self.func = func
        self.kernel = kernel
        self.stride = stride
        self.init_func = init_func
        self.add_layer = []

    def append(self, layer: nn.Module) -> Self:
        self.add_layer.append(layer)
        return self

    def build(self, in_dim: int = None) -> Self:
        self.model = nn.Sequential()

        # lazy init conv layers if no input dim
        conv_type = nn.Conv2d if in_dim else nn.LazyConv2d

        # dims, kernel, stride define conv layers
        for dim, kern, strd in zip(self.dims, self.kernel, self.stride):
            params = (in_dim is not None) * (in_dim,) + (dim, kern)
            layer = self.init_func(conv_type(*params, stride=strd))
            self.model.extend([layer, self.func()])
            conv_type, in_dim = nn.Conv2d, dim

        # flatten final layer outputs
        self.model.append(nn.Flatten(start_dim=-3))

        # append additional layers
        self.model.extend(self.add_layer)

        return self
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)