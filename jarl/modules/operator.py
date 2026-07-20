import torch as th
import torch.nn as nn

from typing import Self

from jarl.envs.gym import SyncGymEnv
from jarl.envs.space import DiscreteSpace, BoxSpace, action_space
from jarl.modules.base import CompositeNet
from jarl.modules.encoder.core import Encoder


class ValueFunction(CompositeNet):
    def __init__(self, head: Encoder, body: nn.Module, foot: nn.Module = None) -> None:
        super().__init__(head, body, foot)
        self._composed = False

    def build(self, env: SyncGymEnv) -> Self:
        return super().build(env)

    def build_composed(self, env: SyncGymEnv, in_dim: int) -> Self:
        if self.foot is None or not hasattr(self.foot, "build"):
            raise TypeError("composed value function requires a buildable foot")
        self.foot.build(in_dim, 1)
        self._composed = True
        return self

    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(x).squeeze(-1)

    def value(
        self, observation: th.Tensor, state: th.Tensor | None = None
    ) -> th.Tensor:
        if self._composed:
            features, _ = self.body_features(observation, state)
            return self.value_from_features(features)
        if state is not None:
            raise ValueError(
                "feed-forward value functions do not accept recurrent state"
            )

        return self(observation)

    def evaluate_values(
        self,
        observation: th.Tensor,
        state: th.Tensor | None = None,
        *,
        reset: th.Tensor | None = None,
    ) -> th.Tensor:
        if self._composed:
            features, _ = self.body_features(observation, state, reset)
            return self.value_from_features(features)
        if reset is not None:
            raise ValueError("feed-forward value functions do not accept reset masks")

        return self.value(observation, state)

    def value_from_features(self, features: th.Tensor) -> th.Tensor:
        return self.foot(features).squeeze(-1)

    def body_features(
        self,
        observation: th.Tensor,
        state:       th.Tensor | None = None,
        reset:       th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor | None]:
        features = self.head(observation)
        if hasattr(self.body, "initial_state"):
            return self.body(features, state, reset)
        if state is not None or reset is not None:
            raise ValueError("stateless value body does not accept state")
        return self.body(features), None


class DiscreteQFunction(CompositeNet):
    def __init__(self, head: Encoder, body: nn.Module, foot: nn.Module = None) -> None:
        super().__init__(head, body, foot)

    def build(self, env: SyncGymEnv) -> Self:
        space = action_space(env)
        assert isinstance(space, DiscreteSpace), (
            "DiscreteQFunction only supports Discrete action"
        )

        return super().build(env, space.numel)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(x).squeeze(-1)


class ContinuousQFunction(CompositeNet):
    def __init__(self, head: Encoder, body: nn.Module, foot: nn.Module = None) -> None:
        super().__init__(head, body, foot)

    def build(self, env: SyncGymEnv) -> Self:
        space = action_space(env)
        assert isinstance(space, BoxSpace), (
            "ContinuousQFunction only supports Box action"
        )

        self.head = self.head if self.head.built else self.head.build(env)
        self.body.build(self.head.feats + space.numel, 1)
        self.foot = self.foot if self.foot else nn.Identity()
        return self

    def forward(self, observation: th.Tensor, action: th.Tensor) -> th.Tensor:
        feats = th.cat((self.head(observation), action), dim=-1)
        return self.foot(self.body(feats)).squeeze(-1)
