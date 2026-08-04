import torch as th
import torch.nn as nn

from typing import Self

from jarl.envs.gym import SyncGymEnv
from jarl.envs.space import DiscreteSpace, BoxSpace, action_space
from jarl.modules.base import CompositeNet
from jarl.modules.encoder.core import Encoder


class Critic(CompositeNet):
    
    def __init__(
        self, 
        foot: Encoder, 
        body: nn.Module, 
        head: nn.Module = None
    ) -> None:
        super().__init__(foot, body, head)
        self._composed = False

    def build(self, env: SyncGymEnv) -> Self:
        return super().build(env)

    def build_composed(self, env: SyncGymEnv, in_dim: int) -> Self:
        if self.head is None or not hasattr(self.head, "build"):
            raise TypeError("composed critic requires a buildable head")
        self.head.build(in_dim, 1)
        self._composed = True
        return self

    def initial_state(self, batch_size: int) -> th.Tensor | None:
        if hasattr(self.body, "initial_state"):
            return self.body.initial_state(batch_size, device=self.device)
        return None

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
                "feed-forward critics do not accept recurrent state"
            )

        return self(observation)

    def evaluate_values(
        self,
        observation: th.Tensor,
        state:       th.Tensor | None = None,
        *,
        reset:       th.Tensor | None = None,
    ) -> th.Tensor:
        if self._composed:
            features, _ = self.body_features(observation, state, reset)
            return self.value_from_features(features)
        if reset is not None:
            raise ValueError("feed-forward critics do not accept reset masks")

        return self.value(observation, state)

    def value_from_features(self, features: th.Tensor) -> th.Tensor:
        return self.head(features).squeeze(-1)

    def body_features(
        self,
        observation: th.Tensor,
        state:       th.Tensor | None = None,
        reset:       th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor | None]:
        features = self.foot(observation)
        if hasattr(self.body, "initial_state"):
            if (
                state is not None
                and state.dtype != features.dtype
                and th.is_autocast_enabled()
            ):
                state = state.to(features.dtype)
            return self.body(features, state, reset)
        if state is not None or reset is not None:
            raise ValueError("stateless critic body does not accept state")
        return self.body(features), None


class DiscreteQFunction(CompositeNet):

    def __init__(
        self,
        foot: Encoder,
        body: nn.Module,
        head: nn.Module = None
    ) -> None:
        super().__init__(foot, body, head)

    def build(self, env: SyncGymEnv) -> Self:
        space = action_space(env)
        assert isinstance(space, DiscreteSpace), (
            "DiscreteQFunction only supports Discrete action"
        )

        return super().build(env, space.numel)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(x).squeeze(-1)


class ContinuousQFunction(CompositeNet):

    def __init__(
        self,
        foot: Encoder,
        body: nn.Module, 
        head: nn.Module = None
    ) -> None:
        super().__init__(foot, body, head)

    def build(self, env: SyncGymEnv) -> Self:
        space = action_space(env)
        assert isinstance(space, BoxSpace), (
            "ContinuousQFunction only supports Box action"
        )

        self.foot = self.foot if self.foot.built else self.foot.build(env)
        self.body.build(self.foot.feats + space.numel, 1)
        self.head = self.head if self.head else nn.Identity()
        return self

    def forward(self, observation: th.Tensor, action: th.Tensor) -> th.Tensor:
        feats = th.cat((self.foot(observation), action), dim=-1)
        return self.head(self.body(feats)).squeeze(-1)
