import torch as th
import torch.nn as nn

from torch.distributions import (
    Distribution,
    Categorical,
    Normal
)

from typing import Self
from abc import ABC, abstractmethod

from jarl.data.records import ActionDecision, Evaluation
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
    def dist(self, obs: th.Tensor) -> Distribution:
        ...

    @abstractmethod
    def action(self, obs: th.Tensor) -> th.Tensor:
        ...

    @abstractmethod
    def sample(self, obs: th.Tensor) -> th.Tensor:
        ...

    def act(
        self,
        obs: th.Tensor,
        state: th.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> ActionDecision:
        if state is not None:
            raise ValueError("feed-forward policies do not accept recurrent state")
        if deterministic:
            return ActionDecision(self.action(obs))
        distribution = self.dist(obs)
        action = self._sample(distribution)
        return ActionDecision(
            action,
            artifacts={"log_prob": self._logprob(distribution, action)},
        )

    def evaluate_actions(
        self,
        obs: th.Tensor,
        action: th.Tensor,
        state: th.Tensor | None = None,
        *,
        reset: th.Tensor | None = None,
    ) -> Evaluation:
        if state is not None:
            raise ValueError("feed-forward policies do not accept recurrent state")
        if reset is not None:
            raise ValueError("feed-forward policies do not accept reset masks")
        distribution = self.dist(obs)
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

    def logprob(self, obs: th.Tensor, act: th.Tensor) -> th.Tensor:
        return self._logprob(self.dist(obs), act)
    
    def entropy(self, obs: th.Tensor) -> th.Tensor:
        return self._entropy(self.dist(obs))

    def forward(self, obs: th.Tensor, sample: bool = True) -> th.Tensor:
        return self.sample(obs) if sample else self.action(obs)
    

class CategoricalPolicy(Policy):

    def __init__(
        self, 
        head: nn.Module, 
        body: nn.Module,
        foot: nn.Module = None
    ) -> None:
        super().__init__(head, body, foot)

    def dist(self, obs: th.Tensor) -> Distribution:
        return Categorical(logits=self.model(obs))
    
    def action(self, obs: th.Tensor) -> th.Tensor:
        return th.argmax(self.model(obs), dim=-1)
    
    def sample(self, obs: th.Tensor) -> th.Tensor:
        return self.dist(obs).sample()


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

    def dist(self, obs: th.Tensor) -> list[Distribution]:
        logits = self.model(obs).split(self.sizes, dim=-1)
        return [Categorical(logits=value) for value in logits]

    def action(self, obs: th.Tensor) -> th.Tensor:
        actions = th.stack(
            [dist.logits.argmax(dim=-1) for dist in self.dist(obs)], dim=-1
        )
        return actions.reshape(*actions.shape[:-1], *self.action_shape)

    def sample(self, obs: th.Tensor) -> th.Tensor:
        return self._sample(self.dist(obs))

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
    
    def dist(self, obs: th.Tensor) -> Distribution:
        loc = self.model(obs)
        std = self.log_std.expand_as(loc).exp()
        return Normal(loc, std)
    
    def _logprob(
        self, distribution: Distribution, action: th.Tensor
    ) -> th.Tensor:
        return super()._logprob(distribution, action).sum(-1)

    def _entropy(self, distribution: Distribution) -> th.Tensor:
        return super()._entropy(distribution).sum(-1)
    
    def action(self, obs: th.Tensor) -> th.Tensor:
        return self.model(obs)
    
    def sample(self, obs: th.Tensor) -> th.Tensor:
        return self.dist(obs).sample()
    

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

    def dist(self, obs: th.Tensor) -> Distribution:
        raise NotImplementedError("dist() not supported")
    
    def action(self, obs: th.Tensor) -> th.Tensor:
        return self.model(obs)
    
    def sample(self, obs: th.Tensor) -> th.Tensor:
        act = self.model(obs)
        return act + th.randn_like(act) * self.scale

    def act(
        self,
        obs: th.Tensor,
        state: th.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> ActionDecision:
        if state is not None:
            raise ValueError("feed-forward policies do not accept recurrent state")
        action = self.action(obs) if deterministic else self.sample(obs)
        return ActionDecision(action)
