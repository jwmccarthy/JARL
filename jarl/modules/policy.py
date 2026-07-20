import torch as th
import torch.nn as nn

from torch.distributions import Distribution, Categorical, Normal

from typing import Self
from abc import ABC, abstractmethod

from jarl.data.records import Evaluation, PolicyOutput
from jarl.envs.gym import SyncGymEnv
from jarl.envs.space import MultiDiscreteSpace, action_space
from jarl.modules.encoder.core import Encoder
from jarl.modules.base import CompositeNet


class Policy(CompositeNet, ABC):
    def __init__(self, head: Encoder, body: nn.Module, foot: nn.Module = None) -> None:
        super().__init__(head, body, foot)

    def build(self, env: SyncGymEnv) -> Self:
        return super().build(env, action_space(env).flat_dim)

    def initial_state(self, batch_size: int):
        if getattr(self, "_composed", False) and hasattr(
            self.body, "initial_state"
        ):
            return self.body.initial_state(batch_size, device=self.device)
        return None

    @abstractmethod
    def dist(self, observation: th.Tensor) -> Distribution: ...

    @abstractmethod
    def action(self, observation: th.Tensor) -> th.Tensor: ...

    @abstractmethod
    def sample(self, observation: th.Tensor) -> th.Tensor: ...

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

    def _logprob(self, distribution: Distribution, action: th.Tensor) -> th.Tensor:
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
        self, head: nn.Module, body: nn.Module, foot: nn.Module = None
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

    def __init__(
        self,
        head:         nn.Module,
        body:         nn.Module,
        foot:         nn.Module = None,
        action_codec=None,
    ) -> None:
        super().__init__(head, body, foot)
        self.action_codec = action_codec
        self._composed = False

    def build(self, env: SyncGymEnv) -> Self:
        output_dim = self._configure_actions(env)
        return CompositeNet.build(self, env, output_dim)

    def build_composed(self, env: SyncGymEnv, in_dim: int) -> Self:
        if self.foot is None or not hasattr(self.foot, "build"):
            raise TypeError("composed policy requires a buildable foot")
        output_dim = self._configure_actions(env)
        self.foot.build(in_dim, output_dim)
        self._composed = True
        return self

    def _configure_actions(self, env: SyncGymEnv) -> int:
        space = action_space(env)
        assert isinstance(space, MultiDiscreteSpace), (
            "MultiCategoricalPolicy only supports MultiDiscrete actions"
        )

        self.action_shape = space.shape
        self.sizes = tuple(int(n) for n in space.nvec.flatten().tolist())
        if self.action_codec is not None:
            if self.action_codec.action_shape != self.action_shape:
                raise ValueError("action codec shape does not match environment actions")
        return sum(self.sizes)

    def initial_state(self, batch_size: int):
        return super().initial_state(batch_size)

    def act(
        self,
        observation: th.Tensor,
        state:       th.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        if not self._composed:
            return super().act(observation, state, deterministic=deterministic)
        features, next_state = self.body_features(observation, state)
        output = self.act_from_features(
            features,
            observation,
            deterministic=deterministic,
        )
        output.next_state = next_state
        return output

    def evaluate_actions(
        self,
        observation: th.Tensor,
        action:      th.Tensor,
        state:       th.Tensor | None = None,
        *,
        reset:       th.Tensor | None = None,
    ) -> Evaluation:
        if not self._composed:
            return super().evaluate_actions(
                observation, action, state, reset=reset
            )
        features, _ = self.body_features(observation, state, reset)
        return self.evaluate_from_features(features, observation, action)

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
            raise ValueError("stateless policy body does not accept state")
        return self.body(features), None

    def act_from_features(
        self,
        features:      th.Tensor,
        observation:   th.Tensor,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        logits = self.foot(features)
        distributions = self._grouped_distributions(logits, observation)
        action = th.empty(
            (*logits.shape[:-1], len(self.sizes)),
            dtype=th.int64,
            device=logits.device,
        )
        for indices, distribution in distributions:
            selected = (
                distribution.logits.argmax(-1)
                if deterministic
                else distribution.sample()
            )
            action[..., list(indices)] = selected
        log_prob = self._grouped_logprob(distributions, action)
        action = action.reshape(*action.shape[:-1], *self.action_shape)
        return PolicyOutput(
            action=action,
            log_prob=log_prob,
        )

    def evaluate_from_features(
        self,
        features:    th.Tensor,
        observation: th.Tensor,
        action:      th.Tensor,
    ) -> Evaluation:
        logits = self.foot(features)
        distributions = self._grouped_distributions(logits, observation)
        return Evaluation(
            log_prob=self._grouped_logprob(distributions, action),
            entropy=sum(
                distribution.entropy().sum(dim=-1)
                for _, distribution in distributions
            ),
        )

    def _grouped_distributions(
        self,
        logits:      th.Tensor,
        observation: th.Tensor,
    ) -> list[tuple[tuple[int, ...], Categorical]]:
        split_logits = logits.split(self.sizes, dim=-1)
        split_masks = None
        if self.action_codec is not None:
            mask = self.action_codec.mask(observation)
            if mask.shape != logits.shape or mask.dtype != th.bool:
                raise ValueError("action codec returned an invalid mask")
            split_masks = mask.split(self.sizes, dim=-1)

        distributions = []
        for size in dict.fromkeys(self.sizes):
            indices = tuple(
                index for index, value in enumerate(self.sizes) if value == size
            )
            grouped = th.stack([split_logits[index] for index in indices], dim=-2)
            if split_masks is not None:
                valid = th.stack(
                    [split_masks[index] for index in indices], dim=-2
                )
                grouped = grouped.masked_fill(~valid, th.finfo(grouped.dtype).min)
            distributions.append((indices, Categorical(logits=grouped)))
        return distributions

    def _grouped_logprob(
        self,
        distributions: list[tuple[tuple[int, ...], Categorical]],
        action:        th.Tensor,
    ) -> th.Tensor:
        action = action.reshape(*action.shape[: -len(self.action_shape)], -1)
        return sum(
            distribution.log_prob(action[..., list(indices)]).sum(dim=-1)
            for indices, distribution in distributions
        )

    def _factorized_distributions(
        self,
        logits:      th.Tensor,
        observation: th.Tensor | None = None,
    ) -> list[Distribution]:
        split_logits = logits.split(self.sizes, dim=-1)
        if self.action_codec is None:
            return [Categorical(logits=value) for value in split_logits]
        if observation is None:
            raise ValueError("masked policy requires observations")
        mask = self.action_codec.mask(observation)
        if mask.shape != logits.shape or mask.dtype != th.bool:
            raise ValueError("action codec returned an invalid mask")
        split_masks = mask.split(self.sizes, dim=-1)
        return [
            Categorical(
                logits=value.masked_fill(~valid, th.finfo(value.dtype).min)
            )
            for value, valid in zip(split_logits, split_masks)
        ]

    def dist(self, observation: th.Tensor) -> list[Distribution]:
        return self._factorized_distributions(self.model(observation), observation)

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
        flat_act = action.reshape(*action.shape[: -len(self.action_shape)], -1)
        logprobs = [
            dist.log_prob(flat_act[..., i]) for i, dist in enumerate(distribution)
        ]
        return th.stack(logprobs, dim=-1).sum(-1)

    def _entropy(self, distribution: list[Distribution]) -> th.Tensor:
        return th.stack([dist.entropy() for dist in distribution], dim=-1).sum(-1)


class DiagonalGaussianPolicy(Policy):
    def __init__(
        self, head: nn.Module, body: nn.Module, foot: nn.Module = None
    ) -> None:
        super().__init__(head, body, foot)

    def build(self, env: SyncGymEnv) -> Self:
        super().build(env)
        log_std = th.zeros((action_space(env).flat_dim))
        self.log_std = nn.Parameter(log_std)
        return self

    def dist(self, observation: th.Tensor) -> Distribution:
        loc = self.model(observation)
        std = self.log_std.expand_as(loc).exp()
        return Normal(loc, std)

    def _logprob(self, distribution: Distribution, action: th.Tensor) -> th.Tensor:
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
        scale: float = 0.1,
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
