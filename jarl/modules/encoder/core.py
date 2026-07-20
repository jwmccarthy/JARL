import torch as th
import torch.nn as nn

from typing import Self

from jarl.envs.gym import SyncGymEnv
from jarl.envs.space import observation_space
from jarl.modules.encoder.base import Encoder
from jarl.modules.utils import init_layer


class FlattenEncoder(Encoder):
    def build(self, env: SyncGymEnv) -> Self:
        super().build(env)
        space = observation_space(env)
        self.start_dim = -len(space.shape)
        self.feats = space.flat_dim
        return self

    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.flatten(x, start_dim=self.start_dim)


class LinearEncoder(Encoder):
    def __init__(
        self,
        out_dim:   int,
        func:      type[nn.Module] = nn.ReLU,
        init_func=init_layer,
    ) -> None:
        super().__init__()
        if out_dim < 1:
            raise ValueError("encoder output dimension must be positive")
        self.out_dim = out_dim
        self.func = func
        self.init_func = init_func

    def build(self, env: SyncGymEnv) -> Self:
        super().build(env)
        space = observation_space(env)
        self.start_dim = -len(space.shape)
        self.model = nn.Sequential(
            self.init_func(nn.Linear(space.flat_dim, self.out_dim)),
            self.func(),
        )
        self.feats = self.out_dim
        return self

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(th.flatten(x, start_dim=self.start_dim))
