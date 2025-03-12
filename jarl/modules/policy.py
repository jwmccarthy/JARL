import torch as th
import torch.nn as nn

from torch.distributions import (
    Distribution,
    Categorical,
    MultivariateNormal
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
        self.covmat = th.eye(env.act_space.flat_dim)
        return self
    
    @lru_cache(maxsize=1)
    def dist(self, obs: th.Tensor) -> Distribution:
        return MultivariateNormal(self.model(obs), self.covmat)
    
    def action(self, obs: th.Tensor) -> th.Tensor:
        return self.model(obs)
    
    def sample(self, obs: th.Tensor) -> th.Tensor:
        return self.dist(obs).sample()
    
    def to(self, device: Device) -> Self:
        self.covmat = self.covmat.to(device)
        return super().to(device)
    

class NoisyContinuousPolicy(Policy):

    def __init__(
        self, 
        head: nn.Module, 
        body: nn.Module,
        foot: nn.Module = None,
        scale: float = 1.0
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