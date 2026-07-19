import torch as th
import torch.nn.functional as F

from jarl.data.batch import TensorBatch
from jarl.transform.base import PrepareContext


class SignRewards:
    requires = frozenset({"rew"})
    produces = frozenset({"rew"})
    replaces = frozenset({"rew"})

    @th.no_grad()
    def __call__(
        self, batch: TensorBatch, context: PrepareContext
    ) -> TensorBatch:
        return batch.replace_fields(rew=batch["rew"].sign())


class DiscriminatorReward:
    requires = frozenset({"obs", "next_obs"})
    replaces = frozenset()

    def __init__(
        self,
        discriminator,
        output_field: str = "imitation_rew",
        from_logits: bool = True,
    ) -> None:
        self.discriminator = discriminator
        self.output_field = output_field
        self.from_logits = from_logits
        self.produces = frozenset({output_field})

    @th.no_grad()
    def __call__(
        self, batch: TensorBatch, context: PrepareContext
    ) -> TensorBatch:
        score = self.discriminator((batch["obs"], batch["next_obs"]))
        reward = F.softplus(-score) if self.from_logits else -score.clamp_min(1e-8).log()
        return batch.with_fields(**{self.output_field: reward})
