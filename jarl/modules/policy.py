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
from jarl.envs.gym import SyncEnv
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

    def build(self, env: SyncEnv) -> Self:
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
    

class DiagonalGaussianPolicy(Policy):

    def __init__(
        self, 
        head: nn.Module, 
        body: nn.Module,
        foot: nn.Module = None
    ) -> None:
        super().__init__(head, body, foot)

    def build(self, env: SyncEnv) -> Self:
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