import torch as th
import torch.nn as nn

from torch.distributions import (
    Distribution,
    Categorical,
    Normal
)

from typing import Self
from abc import ABC, abstractmethod

from jarl.data.records import Evaluation, PolicyOutput
from jarl.envs.gym import SyncGymEnv
from jarl.envs.space import MultiDiscreteSpace
from jarl.modules.encoder.core import Encoder
from jarl.modules.base import CompositeNet


class Policy(CompositeNet, ABC):

    def __init__(
        self,
        head: Encoder,
        body: nn.Module,
        foot: nn.Module = None
    ) -> None:
        super().__init__(head, body, foot)

    def build(self, env: SyncGymEnv) -> Self:
        return super().build(env, env.act_space.flat_dim)

    def initial_state(self, batch_size: int) -> None:
        return None

    @abstractmethod
    def dist(self, observation: th.Tensor) -> Distribution:
        ...

    @abstractmethod
    def action(self, observation: th.Tensor) -> th.Tensor:
        ...

    @abstractmethod
    def sample(self, observation: th.Tensor) -> th.Tensor:
        ...

    def act(
        self,
        observation: th.Tensor,
        state: th.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        if state is not None:
            raise ValueError("feed-forward policies do not accept recurrent state")
        if deterministic:
            return PolicyOutput(self.action(observation))

        distribution = self.dist(observation)
        action = self._sample(distribution)
        return PolicyOutput(
            action=action,
            log_prob=self._logprob(distribution, action),
        )

    def evaluate_actions(
        self,
        observation: th.Tensor,
        action: th.Tensor,
        state: th.Tensor | None = None,
        *,
        reset: th.Tensor | None = None,
    ) -> Evaluation:
        if state is not None:
            raise ValueError("feed-forward policies do not accept recurrent state")
        if reset is not None:
            raise ValueError("feed-forward policies do not accept reset masks")
        distribution = self.dist(observation)
        return Evaluation(
            log_prob=self._logprob(distribution, action),
            entropy=self._entropy(distribution),
        )

    def _sample(self, distribution: Distribution) -> th.Tensor:
        return distribution.sample()

    def _logprob(
        self, distribution: Distribution, action: th.Tensor
    ) -> th.Tensor:
        return distribution.log_prob(action)

    def _entropy(self, distribution: Distribution) -> th.Tensor:
        return distribution.entropy()

    def logprob(self, observation: th.Tensor, action: th.Tensor) -> th.Tensor:
        return self._logprob(self.dist(observation), action)

    def entropy(self, observation: th.Tensor) -> th.Tensor:
        return self._entropy(self.dist(observation))

    def forward(self, observation: th.Tensor, sample: bool = True) -> th.Tensor:
        return self.sample(observation) if sample else self.action(observation)


class CategoricalPolicy(Policy):

    def __init__(
        self,
        head: nn.Module,
        body: nn.Module,
        foot: nn.Module = None
    ) -> None:
        super().__init__(head, body, foot)

    def dist(self, observation: th.Tensor) -> Distribution:
        return Categorical(logits=self.model(observation))

    def action(self, observation: th.Tensor) -> th.Tensor:
        return th.argmax(self.model(observation), dim=-1)

    def sample(self, observation: th.Tensor) -> th.Tensor:
        return self.dist(observation).sample()


class MultiCategoricalPolicy(Policy):
    """Factorized categorical policy for MultiDiscrete action spaces."""

    def build(self, env: SyncGymEnv) -> Self:
        assert isinstance(env.act_space, MultiDiscreteSpace), (
            "MultiCategoricalPolicy only supports MultiDiscrete actions")
        self.action_shape = env.act_space.shape
        self.sizes = tuple(
            int(n) for n in env.act_space.nvec.flatten().tolist()
        )
        return super().build(env)

    def dist(self, observation: th.Tensor) -> list[Distribution]:
        logits = self.model(observation).split(self.sizes, dim=-1)
        return [Categorical(logits=value) for value in logits]

    def action(self, observation: th.Tensor) -> th.Tensor:
        actions = th.stack(
            [dist.logits.argmax(dim=-1) for dist in self.dist(observation)], dim=-1
        )
        return actions.reshape(*actions.shape[:-1], *self.action_shape)

    def sample(self, observation: th.Tensor) -> th.Tensor:
        return self._sample(self.dist(observation))

    def _sample(self, distribution: list[Distribution]) -> th.Tensor:
        actions = th.stack([dist.sample() for dist in distribution], dim=-1)
        return actions.reshape(*actions.shape[:-1], *self.action_shape)

    def _logprob(
        self, distribution: list[Distribution], action: th.Tensor
    ) -> th.Tensor:
        flat_act = action.reshape(*action.shape[:-len(self.action_shape)], -1)
        logprobs = [
            dist.log_prob(flat_act[..., i])
            for i, dist in enumerate(distribution)
        ]
        return th.stack(logprobs, dim=-1).sum(-1)

    def _entropy(self, distribution: list[Distribution]) -> th.Tensor:
        return th.stack(
            [dist.entropy() for dist in distribution], dim=-1
        ).sum(-1)


class DiagonalGaussianPolicy(Policy):

    def __init__(
        self,
        head: nn.Module,
        body: nn.Module,
        foot: nn.Module = None
    ) -> None:
        super().__init__(head, body, foot)

    def build(self, env: SyncGymEnv) -> Self:
        super().build(env)
        log_std = th.zeros((env.act_space.flat_dim))
        self.log_std = nn.Parameter(log_std)
        return self

    def dist(self, observation: th.Tensor) -> Distribution:
        loc = self.model(observation)
        std = self.log_std.expand_as(loc).exp()
        return Normal(loc, std)

    def _logprob(
        self, distribution: Distribution, action: th.Tensor
    ) -> th.Tensor:
        return super()._logprob(distribution, action).sum(-1)

    def _entropy(self, distribution: Distribution) -> th.Tensor:
        return super()._entropy(distribution).sum(-1)

    def action(self, observation: th.Tensor) -> th.Tensor:
        return self.model(observation)

    def sample(self, observation: th.Tensor) -> th.Tensor:
        return self.dist(observation).sample()


class NoisyContinuousPolicy(Policy):

    def __init__(
        self,
        head: nn.Module,
        body: nn.Module,
        foot: nn.Module = None,
        scale: float = 0.1
    ) -> None:
        super().__init__(head, body, foot)
        self.scale = scale

    def dist(self, observation: th.Tensor) -> Distribution:
        raise NotImplementedError("dist() not supported")

    def action(self, observation: th.Tensor) -> th.Tensor:
        return self.model(observation)

    def sample(self, observation: th.Tensor) -> th.Tensor:
        action = self.model(observation)
        return action + th.randn_like(action) * self.scale

    def act(
        self,
        observation: th.Tensor,
        state: th.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        if state is not None:
            raise ValueError("feed-forward policies do not accept recurrent state")
        action = self.action(observation) if deterministic else self.sample(observation)
        return PolicyOutput(action)
