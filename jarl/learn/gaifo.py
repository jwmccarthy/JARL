import torch as th
import torch.nn.functional as F

from jarl.data.batch import TensorBatch
from jarl.learn.update import LossOutput


class GAIFOMinibatches:
    def __init__(self, expert_buffer, batch_size: int, epochs: int = 1) -> None:
        if batch_size < 1 or epochs < 1:
            raise ValueError("batch size and epochs must be positive")
        self.expert_buffer = expert_buffer
        self.batch_size = batch_size
        self.epochs = epochs

    def __call__(self, rollout: TensorBatch):
        flattened = rollout.flatten(0, 1)
        valid = th.ones(len(flattened), dtype=th.bool, device=flattened.device)
        if "learner_mask" in flattened:
            valid &= flattened["learner_mask"].bool()
        if "terminated" in flattened:
            valid &= ~flattened["terminated"].bool()
        if "truncated" in flattened:
            valid &= ~flattened["truncated"].bool()
        agent_transitions = flattened.select("observation", "next_obs")[valid]

        for _ in range(self.epochs):
            yield from self._sample_epoch(agent_transitions)

    def _sample_epoch(self, agent_transitions: TensorBatch):
        indices = th.randperm(
            len(agent_transitions),
            device=agent_transitions.device,
        )

        for start in range(0, len(agent_transitions), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]

            if len(batch_indices) == self.batch_size:
                yield self._build_batch(agent_transitions[batch_indices])

    def _build_batch(self, agent_transitions: TensorBatch) -> TensorBatch:
        expert_transitions = self.expert_buffer.sample(self.batch_size).to(
            agent_transitions.device
        )
        observation = th.cat(
            (
                agent_transitions["observation"],
                expert_transitions["observation"],
            )
        )
        next_obs = th.cat(
            (
                agent_transitions["next_obs"],
                expert_transitions["next_obs"],
            )
        )
        is_agent = th.cat(
            (
                th.ones(self.batch_size, device=agent_transitions.device),
                th.zeros(self.batch_size, device=agent_transitions.device),
            )
        )

        return TensorBatch(
            {
                "observation": observation,
                "next_obs": next_obs,
                "is_agent": is_agent,
            }
        )


class GAIFOLoss:
    def __init__(self, discriminator, from_logits: bool = True) -> None:
        self.discriminator = discriminator
        self.from_logits = from_logits

    def __call__(self, batch: TensorBatch) -> LossOutput:
        discriminator_score = self.discriminator(
            (batch["observation"], batch["next_obs"])
        )
        target = batch["is_agent"]
        loss = (
            F.binary_cross_entropy_with_logits(discriminator_score, target)
            if self.from_logits
            else F.binary_cross_entropy(discriminator_score, target)
        )

        is_agent = target.bool()

        return LossOutput(
            loss,
            {
                "loss": loss.item(),
                "agent_score": discriminator_score[is_agent].mean().item(),
                "expert_score": discriminator_score[~is_agent].mean().item(),
            },
        )
