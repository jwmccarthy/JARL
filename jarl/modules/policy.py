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
from jarl.envs.gym import TorchGymEnv
from jarl.modules.encoder import Encoder
from jarl.modules.network import CompositeNet


class Policy(CompositeNet, ABC):

    def __init__(
        self, 
        head: Encoder, 
        body: nn.Module,
    ) -> None:
        super().__init__(head, body)

    def build(self, env: TorchGymEnv) -> Self:
        super().build(env, env.act_space.flat_dim)

    @abstractmethod
    @lru_cache(maxsize=1)
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
        body: nn.Module
    ) -> None:
        super().__init__(head, body)

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
        body: nn.Module
    ) -> None:
        super().__init__(head, body)

    def build(self, env: TorchGymEnv) -> Self:
        super().build(env)
        self.covmat = th.eye(env.act_space.flat_dim)
        return self

    def dist(self, obs: th.Tensor) -> Distribution:
        return MultivariateNormal(self.model(obs), self.covmat)
    
    def action(self, obs: th.Tensor) -> th.Tensor:
        return self.model(obs)
    
    def sample(self, obs: th.Tensor) -> th.Tensor:
        return self.dist(obs).sample()
    
    def to(self, device: Device) -> Self:
        self.covmat = self.covmat.to(device)
        return super().to(device)