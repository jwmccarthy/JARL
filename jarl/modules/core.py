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
        self.model = nn.Sequential()

        # dims define hidden layers
        for next_dim in self.dims:
            self.model.extend([nn.Linear(in_dim, next_dim), self.func()])
            in_dim = next_dim

        # output linear layer
        self.model.append(nn.Linear(in_dim, out_dim))

        return self

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)