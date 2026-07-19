import torch as th
import torch.nn.functional as F

from jarl.data.batch import TensorBatch
from jarl.learn.update import LossOutput


class GAIFOMinibatches:
    def __init__(self, expert_buffer, batch_size: int, epochs: int = 1) -> None:
        self.expert_buffer = expert_buffer
        self.batch_size = batch_size
        self.epochs = epochs

    def __call__(self, rollout: TensorBatch):
        agent = rollout.select("observation", "next_obs").flatten(0, 1)

        for _ in range(self.epochs):
            indices = th.randperm(len(agent), device=agent.device)
            for left in range(0, len(agent), self.batch_size):
                selected = indices[left : left + self.batch_size]
                if len(selected) != self.batch_size:
                    continue

                agent_batch = agent[selected]
                expert_batch = self.expert_buffer.sample(self.batch_size).to(
                    agent.device
                )
                yield TensorBatch(
                    {
                        "observation": th.cat(
                            (agent_batch["observation"], expert_batch["observation"])
                        ),
                        "next_obs": th.cat(
                            (agent_batch["next_obs"], expert_batch["next_obs"])
                        ),
                        "is_agent": th.cat(
                            (
                                th.ones(self.batch_size, device=agent.device),
                                th.zeros(self.batch_size, device=agent.device),
                            )
                        ),
                    }
                )


class GAIFOLoss:
    def __init__(self, discriminator, from_logits: bool = True) -> None:
        self.discriminator = discriminator
        self.from_logits = from_logits

    def __call__(self, batch: TensorBatch) -> LossOutput:
        score = self.discriminator((batch["observation"], batch["next_obs"]))
        target = batch["is_agent"]
        loss = (
            F.binary_cross_entropy_with_logits(score, target)
            if self.from_logits
            else F.binary_cross_entropy(score, target)
        )

        agent = target.bool()
        return LossOutput(
            loss,
            {
                "loss": loss.item(),
                "agent_score": score[agent].mean().item(),
                "expert_score": score[~agent].mean().item(),
            },
        )
