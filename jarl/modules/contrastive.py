"""Residual goal-conditioned networks for contrastive reinforcement learning."""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.checkpoint import checkpoint


def _linear(in_features: int, out_features: int) -> nn.Linear:
    """LeCun-uniform initialization with zero biases."""
    layer = nn.Linear(in_features, out_features)
    bound = 1.0 / math.sqrt(in_features)
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.zeros_(layer.bias)
    return layer


def _dense_unit(in_features: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        _linear(in_features, out_features),
        nn.LayerNorm(out_features, eps=1e-6),
        nn.SiLU(),
    )


class ResidualBlock(nn.Module):
    """Four dense/LayerNorm/Swish units followed by an identity connection."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(*(_dense_unit(width, width) for _ in range(4)))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x + self.layers(x)


class ResidualNetwork(nn.Module):
    """Depth counts layers in residual blocks, excluding the stem and readout."""

    def __init__(
        self, in_features: int, out_features: int, width: int, depth: int,
        *, checkpoint_activations: bool = False,
    ) -> None:
        super().__init__()
        if width < 1 or depth < 4 or depth % 4:
            raise ValueError("network width must be positive and depth a multiple of four")
        self.stem = _dense_unit(in_features, width)
        self.blocks = nn.ModuleList(ResidualBlock(width) for _ in range(depth // 4))
        self.head = _linear(width, out_features)
        self.checkpoint_activations = checkpoint_activations

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            if self.checkpoint_activations and self.training and th.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.head(x)


def one_hot_actions(actions: th.Tensor, nvec: tuple[int, ...]) -> th.Tensor:
    if actions.shape[-1] != len(nvec):
        raise ValueError("actions must match the factorized action space")
    return th.cat([
        F.one_hot(actions[..., i].long(), n).float() for i, n in enumerate(nvec)
    ], dim=-1)


def action_distributions(
    logits: th.Tensor, mask: th.Tensor, nvec: tuple[int, ...],
) -> tuple[Categorical, ...]:
    if logits.shape != mask.shape or logits.shape[-1] != sum(nvec):
        raise ValueError("action logits and mask must match the action space")
    return tuple(
        Categorical(logits=part.masked_fill(~valid, -th.inf))
        for part, valid in zip(logits.split(nvec, dim=-1), mask.split(nvec, dim=-1))
    )


class GoalActor(nn.Module):
    """Residual masked MultiDiscrete policy conditioned on an achieved-state goal."""

    def __init__(
        self, observation_size: int, goal_size: int, width: int, depth: int,
        nvec: tuple[int, ...], action_codec,
        *, checkpoint_activations: bool = False,
    ) -> None:
        super().__init__()
        if not nvec or any(n < 2 for n in nvec):
            raise ValueError("categorical action factors must contain at least two choices")
        self.observation_size = observation_size
        self.goal_size = goal_size
        self.nvec = tuple(nvec)
        self.action_codec = action_codec
        self.network = ResidualNetwork(
            observation_size + goal_size, sum(nvec), width, depth,
            checkpoint_activations=checkpoint_activations,
        )

    @property
    def device(self) -> th.device:
        return next(self.parameters()).device

    def forward(self, observation: th.Tensor, goal: th.Tensor) -> th.Tensor:
        return self.network(th.cat((observation, goal), dim=-1))

    def distributions(
        self, observation: th.Tensor, goal: th.Tensor,
    ) -> tuple[Categorical, ...]:
        return action_distributions(
            self(observation, goal), self.action_codec.mask(observation), self.nvec,
        )

    @th.no_grad()
    def act(
        self, observation: th.Tensor, goal: th.Tensor, *, deterministic: bool = False,
    ) -> th.Tensor:
        distributions = self.distributions(observation, goal)
        return th.stack([
            d.logits.argmax(dim=-1) if deterministic else d.sample()
            for d in distributions
        ], dim=-1).to(th.int32)


class ContrastiveCritic(nn.Module):
    """State-action and goal encoders scored by negative Euclidean distance."""

    def __init__(
        self, observation_size: int, goal_size: int, width: int, depth: int,
        embedding_size: int, nvec: tuple[int, ...],
        *, checkpoint_activations: bool = False,
    ) -> None:
        super().__init__()
        self.state_action = ResidualNetwork(
            observation_size + sum(nvec), embedding_size, width, depth,
            checkpoint_activations=checkpoint_activations,
        )
        self.goal = ResidualNetwork(
            goal_size, embedding_size, width, depth,
            checkpoint_activations=checkpoint_activations,
        )

    def q(
        self, observation: th.Tensor, action_one_hot: th.Tensor,
        goal: th.Tensor,
    ) -> th.Tensor:
        sa = self.state_action(th.cat((observation, action_one_hot), dim=-1))
        return -th.linalg.vector_norm(sa - self.goal(goal), dim=-1)
