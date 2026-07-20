import torch as th

from typing import Self

from jarl.envs.gym import SyncGymEnv
from jarl.envs.space import observation_space
from jarl.modules.encoder.base import Encoder


class FlattenEncoder(Encoder):
    def build(self, env: SyncGymEnv) -> Self:
        super().build(env)
        space = observation_space(env)
        self.start_dim = -len(space.shape)
        self.feats = space.flat_dim
        return self

    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.flatten(x, start_dim=self.start_dim)
