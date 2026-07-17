import torch as th
import torch.nn as nn

from torch.distributions import (
    Distribution,
    Categorical,
    Normal
)

from typing import Self
from functools import lru_cache
from abc import ABC, abstractmethod

from jarl.data.types import Device
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

    @abstractmethod
    def dist(self, obs: th.Tensor) -> Distribution:
        ...

    @abstractmethod
    def action(self, obs: th.Tensor) -> th.Tensor:
        ...

    @abstractmethod
    def sample(self, obs: th.Tensor) -> th.Tensor:
        ...

    def logprob(self, obs: th.Tensor, act: th.Tensor) -> th.Tensor:
        return self.dist(obs).log_prob(act)
    
    def entropy(self, obs: th.Tensor) -> th.Tensor:
        return self.dist(obs).entropy()

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

    @lru_cache(maxsize=1)
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
        actions = th.stack([dist.sample() for dist in self.dist(obs)], dim=-1)
        return actions.reshape(*actions.shape[:-1], *self.action_shape)

    def logprob(self, obs: th.Tensor, act: th.Tensor) -> th.Tensor:
        flat_act = act.reshape(*act.shape[:-len(self.action_shape)], -1)
        logprobs = [
            dist.log_prob(flat_act[..., i])
            for i, dist in enumerate(self.dist(obs))
        ]
        return th.stack(logprobs, dim=-1).sum(-1)

    def entropy(self, obs: th.Tensor) -> th.Tensor:
        return th.stack(
            [dist.entropy() for dist in self.dist(obs)], dim=-1
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
    
    @lru_cache(maxsize=1)
    def dist(self, obs: th.Tensor) -> Distribution:
        loc = self.model(obs)
        std = self.log_std.expand_as(loc).exp()
        return Normal(loc, std)
    
    def logprob(self, obs: th.Tensor, act: th.Tensor) -> th.Tensor:
        return super().logprob(obs, act).sum(-1)
    
    def entropy(self, obs: th.Tensor) -> th.Tensor:
        return super().entropy(obs).sum(-1)
    
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
