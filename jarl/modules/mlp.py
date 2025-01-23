import torch.nn as nn
from torch import Tensor

from typing import List, Self


class MLP(nn.Module):

    model: nn.Module

    def __init__(
        self, 
        dims: List[int] = [64, 64],
        func: nn.Module = nn.ReLU
    ) -> None:
        super().__init__()
        self.dims = dims  # hidden layer dims
        self.func = func  # activation function

    def build(self, in_dim: int, out_dim: int) -> Self:
        modules = []

        # dims define hidden layers
        for next_dim in self.dims:
            modules.extend([nn.Linear(in_dim, next_dim), self.func()])
            in_dim = next_dim

        # output linear layer
        modules.append(nn.Linear(in_dim, out_dim))

        # wrap modules in sequential
        self.model =  nn.Sequential(*modules)

        return self

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)