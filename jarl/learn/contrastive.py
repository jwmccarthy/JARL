"""Entropy-regularized contrastive goal-reaching for masked discrete actions."""

import math
from collections.abc import Callable

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from jarl.modules.contrastive import (
    ContrastiveCritic, GoalActor, one_hot_actions,
)
from jarl.store.goal_replay import ContrastiveBatch


class ContrastiveLearner:
    """Actor/critic/temperature updates with diagonal-positive InfoNCE."""

    def __init__(
        self, observation_size: int, goal_size: int, action_codec, nvec: tuple[int, ...],
        *, actor_width: int, actor_depth: int, critic_width: int, critic_depth: int,
        embedding_size: int, actor_lr: float, critic_lr: float, alpha_lr: float,
        entropy_target_fraction: float, logsumexp_penalty: float,
        device: str | th.device, checkpoint_activations: bool = True,
        max_grad_norm: float | None = None,
    ) -> None:
        self.device = th.device(device)
        self.observation_size = observation_size
        self.nvec = tuple(nvec)
        self.entropy_target = entropy_target_fraction * sum(math.log(n) for n in nvec)
        self.logsumexp_penalty = logsumexp_penalty
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("gradient norm limit must be positive")
        self.max_grad_norm = max_grad_norm
        self.actor = GoalActor(
            observation_size, goal_size, actor_width, actor_depth,
            nvec, action_codec,
            checkpoint_activations=checkpoint_activations and actor_depth >= 64,
        ).to(self.device)
        self.critic = ContrastiveCritic(
            observation_size, goal_size, critic_width, critic_depth,
            embedding_size, nvec,
            checkpoint_activations=checkpoint_activations and critic_depth >= 64,
        ).to(self.device)
        self.log_alpha = nn.Parameter(th.zeros((), device=self.device))
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_optimizer = th.optim.Adam([self.log_alpha], lr=alpha_lr)

    def act(
        self, observation: th.Tensor, goal: th.Tensor, *, deterministic: bool = False,
    ) -> th.Tensor:
        return self.actor.act(observation, goal, deterministic=deterministic)

    def update(self, batch: ContrastiveBatch) -> dict[str, th.Tensor]:
        observation, goal = batch.observation, batch.goal
        if any(
            value.device != self.device
            for value in (observation, goal, batch.action, batch.steps_to_goal)
        ):
            raise ValueError("contrastive minibatches must stay on the learner device")

        self.actor_optimizer.zero_grad(set_to_none=True)
        # Propagate the Q derivative through hard Gumbel-Softmax actions without
        # accumulating gradients on the frozen critic's parameters.
        self.critic.requires_grad_(False)
        try:
            distributions = self.actor.distributions(observation, goal)
            action = th.cat([
                F.gumbel_softmax(d.logits, tau=1.0, hard=True, dim=-1)
                for d in distributions
            ], dim=-1)
            entropy = sum(d.entropy() for d in distributions)
            q = self.critic.q(observation, action, goal)
            alpha = self.log_alpha.exp().detach()
            actor_loss = -(q + alpha * entropy).mean() if self.entropy_target else -q.mean()
            actor_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
        finally:
            self.critic.requires_grad_(True)

        if self.entropy_target:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss = self.log_alpha.exp() * (entropy.detach().mean() - self.entropy_target)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.critic_optimizer.zero_grad(set_to_none=True)
        state_action = self.critic.state_action(th.cat((
            observation, one_hot_actions(batch.action, self.nvec),
        ), dim=-1))
        goal_embedding = self.critic.goal(goal)
        logits = -th.cdist(state_action, goal_embedding)
        logsumexp = th.logsumexp(logits + 1e-6, dim=-1)
        critic_loss = (
            F.cross_entropy(logits, th.arange(len(goal), device=self.device))
            + self.logsumexp_penalty * logsumexp.square().mean()
        )
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "q": q.detach().mean(),
            "entropy": entropy.detach().mean(),
            "alpha": self.log_alpha.detach().exp(),
            "retrieval_accuracy": (
                logits.detach().argmax(dim=-1)
                == th.arange(len(goal), device=self.device)
            ).float().mean(),
            "future_steps": batch.steps_to_goal.float().mean(),
        }


class ContrastiveUpdate:
    """JARL Algorithm stage: several GPU minibatches per collection block."""

    def __init__(
        self, learner: ContrastiveLearner, batch_size: int, steps: int,
        *, steps_for_update: Callable[[], int] | None = None,
    ) -> None:
        if batch_size < 2 or steps < 1:
            raise ValueError("contrastive update needs a batch of two and positive steps")
        self.learner = learner
        self.batch_size = batch_size
        self.steps = steps
        self.steps_for_update = steps_for_update
        self.gradient_steps = 0
        self._progress_callback = None
        self.last_metrics: dict[str, float] = {}

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback

    def run(self, replay) -> tuple[object, dict[str, dict[str, float]]]:
        steps = self.steps_for_update() if self.steps_for_update is not None else self.steps
        if steps < 1:
            raise ValueError("contrastive update needs at least one optimizer step")
        callback = self._progress_callback
        if callback is not None:
            callback.start_activity(steps, "CRL optimizer")
        totals: dict[str, th.Tensor] = {}
        try:
            for _ in range(steps):
                for name, value in self.learner.update(replay.sample(self.batch_size)).items():
                    totals[name] = totals.get(name, 0) + value
                self.gradient_steps += 1
                if callback is not None:
                    callback.advance_activity()
        finally:
            if callback is not None:
                callback.finish_activity()
        self.last_metrics = {
            **{name: (total / steps).item() for name, total in totals.items()},
            "gradient_steps": float(self.gradient_steps),
        }
        if not all(math.isfinite(value) for value in self.last_metrics.values()):
            raise FloatingPointError("non-finite contrastive update metrics")
        return replay, {"CRL": self.last_metrics}
