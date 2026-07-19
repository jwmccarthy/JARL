from dataclasses import dataclass

import torch as th

from jarl.data.batch import TensorBatch
from jarl.sample.rollout import SequenceBatch
from jarl.store.rollout import Rollout
from jarl.transform.base import PrepareContext, apply_transforms


@dataclass(frozen=True)
class PPOConfig:
    clip: float = 0.2
    value_clip: float | None = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    normalize_advantage: bool = True


class PPOOptimizer:
    def __init__(
        self,
        policy,
        value_function,
        minibatches,
        optimizer_step,
        config: PPOConfig = PPOConfig(),
    ) -> None:
        self.policy = policy
        self.value_function = value_function
        self.minibatches = minibatches
        self.optimizer_step = optimizer_step
        self.config = config

    def update(self, data: TensorBatch) -> dict[str, float]:
        totals: dict[str, float] = {}
        count = 0
        for sample in self.minibatches(data):
            metrics = self._update_minibatch(sample)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
        if count == 0:
            raise RuntimeError("PPO sampler produced no minibatches")
        self.optimizer_step.advance_scheduler()
        versioned = {id(module): module for module in (self.policy, self.value_function)}
        for module in versioned.values():
            module.increment_version()
        return {key: value / count for key, value in totals.items()}

    def _update_minibatch(
        self, sample: TensorBatch | SequenceBatch
    ) -> dict[str, float]:
        if isinstance(sample, SequenceBatch):
            batch = sample.steps
            state = sample.initial_state
            reset = sample.reset
            valid = sample.valid
        else:
            batch = sample
            state = reset = None
            valid = th.ones_like(batch["adv"], dtype=th.bool)

        evaluation = self.policy.evaluate_actions(
            batch["obs"],
            batch["act"],
            state,
            reset=reset,
        )
        value = self.value_function.evaluate_values(
            batch["obs"],
            state,
            reset=reset,
        )
        advantage = batch["adv"][valid]
        if self.config.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (
                advantage.std(unbiased=False) + 1e-8
            )

        log_ratio = (
            evaluation.log_prob[valid] - batch["behavior_log_prob"][valid]
        )
        ratio = log_ratio.exp()
        policy_loss = -th.minimum(
            advantage * ratio,
            advantage * ratio.clamp(
                1 - self.config.clip,
                1 + self.config.clip,
            ),
        ).mean()

        predicted_value = value[valid]
        target = batch["ret"][valid]
        value_loss = (predicted_value - target).pow(2)
        if self.config.value_clip is not None:
            old_value = batch["baseline_value"][valid]
            clipped = old_value + (predicted_value - old_value).clamp(
                -self.config.value_clip,
                self.config.value_clip,
            )
            value_loss = th.maximum(value_loss, (clipped - target).pow(2))
        value_loss = 0.5 * value_loss.mean()
        entropy = evaluation.entropy[valid].mean()
        loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy
        )
        self.optimizer_step(loss)

        with th.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
        return {
            "policy_loss": policy_loss.item(),
            "critic_loss": value_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": approx_kl.item(),
        }


class PPOLearner:
    def __init__(self, transforms, optimizer: PPOOptimizer) -> None:
        self.transforms = tuple(transforms)
        self.optimizer = optimizer

    def update(self, rollout: Rollout) -> dict[str, dict[str, float]]:
        policy_versions = rollout.steps["policy_version"].unique()
        if (
            policy_versions.numel() != 1
            or policy_versions.item() != self.optimizer.policy.version
        ):
            raise RuntimeError("rollout was not collected by the current policy version")
        prepared = apply_transforms(
            rollout.steps,
            self.transforms,
            PrepareContext(rollout),
        )
        value_versions = prepared["value_version"].unique()
        if (
            value_versions.numel() != 1
            or value_versions.item() != self.optimizer.value_function.version
        ):
            raise RuntimeError("rollout was not valued by the current value version")
        return {"PPO": self.optimizer.update(prepared)}
