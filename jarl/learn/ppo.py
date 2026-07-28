from dataclasses import dataclass

import torch as th

from jarl.data.batch import TensorBatch
from jarl.learn.update import LossOutput
from jarl.sample.rollout import SequenceBatch


@dataclass(frozen=True)
class PPOConfig:
    clip:                float = 0.2
    value_clip:          float | None = 0.2
    value_coef:          float = 0.5
    entropy_coef:        float = 0.01
    normalize_advantage: bool = True
    bf16:                bool = False
    action_names:        tuple[str, ...] | None = None


class PPOLoss:
    def __init__(
        self,
        policy,
        critic,
        config: PPOConfig = PPOConfig(),
    ) -> None:
        self.policy = policy
        self.critic = critic
        self.config = config

    def __call__(self, sample: TensorBatch | SequenceBatch) -> LossOutput:
        batch, state, critic_state, reset, valid = self._unpack_sample(sample)
        use_bf16 = self.config.bf16 and batch.device.type == "cuda"

        with th.autocast(
            device_type=batch.device.type,
            dtype=th.bfloat16,
            enabled=use_bf16,
        ):
            if self._shares_trunk():
                features, _ = self.policy.body_features(
                    batch["observation"],
                    state,
                    reset,
                )
                evaluation = self.policy.evaluate_from_features(
                    features,
                    batch["observation"],
                    batch["action"],
                )
                value = self.critic.value_from_features(features)
            else:
                evaluation = self.policy.evaluate_actions(
                    batch["observation"],
                    batch["action"],
                    state,
                    reset=reset,
                )
                value = evaluation.value
                if value is None:
                    value = self.critic.evaluate_values(
                        batch["observation"],
                        critic_state,
                        reset=reset,
                    )

        # Keep PPO's ratios, reductions, and value loss in FP32.
        evaluation.log_prob = evaluation.log_prob.float()
        evaluation.entropy = evaluation.entropy.float()
        value = value.float()

        advantage = self._normalize_advantage(batch["advantage"][valid])
        log_ratio = evaluation.log_prob[valid] - batch["old_log_prob"][valid]
        ratio = log_ratio.exp()

        policy_loss = self._policy_loss(advantage, ratio)
        value_loss = self._value_loss(value[valid], batch, valid)
        entropy = evaluation.entropy[valid].mean()
        loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy
        )

        with th.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()

        metrics = {
            "policy_loss": policy_loss,
            "critic_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
        }
        factor_entropy = evaluation.extras.get("factor_entropy")
        if factor_entropy is not None:
            action_shape = len(self.policy.action_shape)
            action = batch["action"].reshape(
                *batch["action"].shape[:-action_shape], -1
            )
            if factor_entropy.shape[:-1] != valid.shape:
                raise ValueError("factor entropy shape does not match PPO validity mask")
            names = self.config.action_names or tuple(
                str(index) for index in range(factor_entropy.shape[-1])
            )
            if len(names) != factor_entropy.shape[-1]:
                raise ValueError("action names do not match policy action factors")
            for index, name in enumerate(names):
                metrics[f"entropy_{name}"] = factor_entropy[..., index][valid].mean()
                for value in range(self.policy.sizes[index]):
                    metrics[f"action_{name}_{value}_rate"] = (
                        action[..., index][valid].eq(value).float().mean()
                    )

        return LossOutput(loss, metrics)

    def _shares_trunk(self) -> bool:
        return (
            self.policy.head is self.critic.head
            and self.policy.body is self.critic.body
            and hasattr(self.policy, "body_features")
            and hasattr(self.policy, "evaluate_from_features")
            and hasattr(self.critic, "value_from_features")
        )

    @staticmethod
    def _unpack_sample(sample: TensorBatch | SequenceBatch):
        if isinstance(sample, SequenceBatch):
            critic_state = sample.initial_critic_state
            if critic_state is None:
                critic_state = sample.initial_state
            return (
                sample.steps,
                sample.initial_state,
                critic_state,
                sample.reset,
                sample.valid,
            )

        valid = th.ones_like(sample["advantage"], dtype=th.bool)
        return sample, None, None, None, valid

    def _normalize_advantage(self, advantage: th.Tensor) -> th.Tensor:
        if self.config.normalize_advantage:
            return (advantage - advantage.mean()) / (
                advantage.std(unbiased=False) + 1e-8
            )
        return advantage

    def _policy_loss(
        self,
        advantage: th.Tensor,
        ratio: th.Tensor,
    ) -> th.Tensor:
        return -th.minimum(
            advantage * ratio,
            advantage
            * ratio.clamp(
                1 - self.config.clip,
                1 + self.config.clip,
            ),
        ).mean()

    def _value_loss(
        self,
        predicted_value: th.Tensor,
        batch: TensorBatch,
        valid: th.Tensor,
    ) -> th.Tensor:
        target = batch["returns"][valid]
        value_loss = (predicted_value - target).pow(2)

        if self.config.value_clip is not None:
            old_value = batch["baseline_value"][valid]
            clipped = old_value + (predicted_value - old_value).clamp(
                -self.config.value_clip,
                self.config.value_clip,
            )
            value_loss = th.maximum(value_loss, (clipped - target).pow(2))

        return 0.5 * value_loss.mean()
