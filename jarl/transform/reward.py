import torch as th
import torch.nn.functional as F

from jarl.data.batch import TensorBatch
from jarl.transform.base import PrepareContext


class SignRewards:
    @th.no_grad()
    def __call__(
        self, batch: TensorBatch, context: PrepareContext
    ) -> TensorBatch:
        return batch.replace_fields(reward=batch["reward"].sign())


class TeamSpirit:
    def __init__(
        self,
        num_matches: int,
        team_sizes: tuple[int, int],
        spirit: float,
    ) -> None:
        if num_matches < 1 or any(size < 1 for size in team_sizes):
            raise ValueError("team dimensions must be positive")
        if not 0.0 <= spirit <= 1.0:
            raise ValueError("team spirit must be between zero and one")

        self.num_matches = num_matches
        self.team_sizes = team_sizes
        self.players_per_match = sum(team_sizes)
        self.spirit = spirit

    @th.no_grad()
    def __call__(
        self, batch: TensorBatch, context: PrepareContext
    ) -> TensorBatch:
        reward = batch["reward"]
        expected = self.num_matches * self.players_per_match
        if reward.shape[-1] != expected:
            raise ValueError(
                f"expected {expected} actor rewards, got {reward.shape[-1]}"
            )

        grouped = reward.reshape(
            *reward.shape[:-1], self.num_matches, self.players_per_match
        )
        mixed = grouped.clone()
        left = 0
        for size in self.team_sizes:
            right = left + size
            team_reward = grouped[..., left:right]
            team_mean = team_reward.mean(dim=-1, keepdim=True)
            mixed[..., left:right] = (
                (1.0 - self.spirit) * team_reward + self.spirit * team_mean
            )
            left = right

        return batch.replace_fields(reward=mixed.reshape_as(reward))


class DiscriminatorReward:
    def __init__(
        self,
        discriminator,
        output_field: str = "imitation_reward",
        from_logits: bool = True,
        mask_terminal: bool = False,
        reward_type: str = "softplus",
    ) -> None:
        if reward_type not in ("softplus", "negative_logit"):
            raise ValueError("unknown discriminator reward type")
        if reward_type == "negative_logit" and not from_logits:
            raise ValueError("negative logit reward requires logits")
        self.discriminator = discriminator
        self.output_field = output_field
        self.from_logits = from_logits
        self.mask_terminal = mask_terminal
        self.reward_type = reward_type

    @th.no_grad()
    def __call__(
        self, batch: TensorBatch, context: PrepareContext
    ) -> TensorBatch:
        score = self.discriminator(
            (batch["observation"], batch["next_obs"])
        )
        if self.reward_type == "negative_logit":
            reward = -score
        else:
            reward = (
                F.softplus(-score)
                if self.from_logits
                else -score.clamp_min(1e-8).log()
            )
        if self.mask_terminal:
            terminal = batch["terminated"].bool() | batch["truncated"].bool()
            reward = reward.masked_fill(terminal, 0.0)
        return batch.with_fields(**{self.output_field: reward})
