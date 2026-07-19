import torch as th
import torch.nn as nn

from typing import Self

from jarl.envs.gym import SyncGymEnv
from jarl.envs.space import DiscreteSpace, BoxSpace
from jarl.modules.base import CompositeNet
from jarl.modules.encoder.core import Encoder


class ValueFunction(CompositeNet):

    def __init__(
        self,
        head: Encoder,
        body: nn.Module,
        foot: nn.Module = None
    ) -> None:
        super().__init__(head, body, foot)

    def build(self, env: SyncGymEnv) -> Self:
        return super().build(env)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(x).squeeze(-1)

    def value(
        self, observation: th.Tensor, state: th.Tensor | None = None
    ) -> th.Tensor:
        if state is not None:
            raise ValueError("feed-forward value functions do not accept recurrent state")

        return self(observation)

    def evaluate_values(
        self,
        observation: th.Tensor,
        state: th.Tensor | None = None,
        *,
        reset: th.Tensor | None = None,
    ) -> th.Tensor:
        if reset is not None:
            raise ValueError("feed-forward value functions do not accept reset masks")

        return self.value(observation, state)


class DiscreteQFunction(CompositeNet):

    def __init__(
        self,
        head: Encoder,
        body: nn.Module,
        foot: nn.Module = None
    ) -> None:
        super().__init__(head, body, foot)

    def build(self, env: SyncGymEnv) -> Self:
        assert isinstance(env.act_space, DiscreteSpace), (
            "DiscreteQFunction only supports Discrete action")

        return super().build(env, env.act_space.numel)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(x).squeeze(-1)


class ContinuousQFunction(CompositeNet):

    def __init__(
        self,
        head: Encoder,
        body: nn.Module,
        foot: nn.Module = None
    ) -> None:
        super().__init__(head, body, foot)

    def build(self, env: SyncGymEnv) -> Self:
        assert isinstance(env.act_space, BoxSpace), (
            "ContinuousQFunction only supports Box action")

        self.head = self.head if self.head.built else self.head.build(env)
        self.body.build(self.head.feats + env.act_space.numel, 1)
        self.foot = self.foot if self.foot else nn.Identity()
        return self

    def forward(self, observation: th.Tensor, action: th.Tensor) -> th.Tensor:
        feats = th.cat((self.head(observation), action), dim=-1)
        return self.foot(self.body(feats)).squeeze(-1)
