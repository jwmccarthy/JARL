import torch as th
import torch.nn.functional as F

from jarl.data.batch import TensorBatch
from jarl.learn.optim import OptimizerStep


class TrainDiscriminator:
    produces = frozenset()

    def __init__(
        self,
        source: str,
        expert_source,
        discriminator,
        optimizer_step: OptimizerStep,
        batch_size: int,
        epochs: int = 1,
        from_logits: bool = True,
        section: str = "Discriminator",
    ) -> None:
        self.source = source
        self.expert_source = expert_source
        self.discriminator = discriminator
        self.optimizer_step = optimizer_step
        self.batch_size = batch_size
        self.epochs = epochs
        self.from_logits = from_logits
        self.section = section
        self.requires = frozenset({source})

    def run(self, workspace) -> None:
        rollout = workspace.require(self.source)
        agent = rollout.steps.select("obs", "next_obs").flatten(0, 1)
        losses = []
        for _ in range(self.epochs):
            indices = th.randperm(len(agent), device=agent.device)
            for left in range(0, len(agent), self.batch_size):
                selected = indices[left : left + self.batch_size]
                if len(selected) != self.batch_size:
                    continue
                agent_batch = agent[selected]
                expert_batch: TensorBatch = self.expert_source.sample(self.batch_size)
                expert_batch = expert_batch.to(agent.device)
                agent_score = self.discriminator(
                    (agent_batch["obs"], agent_batch["next_obs"])
                )
                expert_score = self.discriminator(
                    (expert_batch["obs"], expert_batch["next_obs"])
                )
                score = th.cat((agent_score, expert_score))
                target = th.cat((th.ones_like(agent_score), th.zeros_like(expert_score)))
                loss = (
                    F.binary_cross_entropy_with_logits(score, target)
                    if self.from_logits
                    else F.binary_cross_entropy(score, target)
                )
                self.optimizer_step(loss)
                losses.append(loss.item())
        if not losses:
            raise RuntimeError("discriminator stage produced no minibatches")
        workspace.add_metrics(
            self.section,
            {"loss": sum(losses) / len(losses)},
        )
