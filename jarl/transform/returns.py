import torch as th

from jarl.data.batch import TensorBatch
from jarl.transform.base import PrepareContext


def discounted_suffix_sum(
    value:     th.Tensor,
    continues: th.Tensor,
    discount:  float,
) -> th.Tensor:
    if value.shape != continues.shape:
        raise ValueError("value and continuation masks must have matching shapes")
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be between zero and one")
    if discount == 0.0:
        return value.clone()

    time = len(value)
    shape = (time,) + (1,) * (value.ndim - 1)
    powers = th.pow(
        th.as_tensor(discount, dtype=value.dtype, device=value.device),
        th.arange(time, dtype=value.dtype, device=value.device),
    ).view(shape)
    weighted = value * powers
    suffix = th.flip(th.cumsum(th.flip(weighted, dims=(0,)), dim=0), dims=(0,))

    positions = th.arange(time, device=value.device).view(shape).expand_as(value)
    boundaries = th.where(continues, time, positions)
    boundaries[-1] = time - 1
    segment_end = th.flip(
        th.cummin(th.flip(boundaries, dims=(0,)), dim=0).values,
        dims=(0,),
    )
    padded = th.cat((suffix, th.zeros_like(suffix[:1])), dim=0)
    after_segment = padded.gather(0, segment_end + 1)

    return (suffix - after_segment) / powers


class GAE:
    def __init__(
        self,
        gamma:        float = 0.99,
        lambda_:      float = 0.95,
        reward_field: str = "reward",
    ) -> None:
        self.gamma = gamma
        self.lambda_ = lambda_
        self.reward_field = reward_field

    @th.no_grad()
    def __call__(self, batch: TensorBatch, context: PrepareContext) -> TensorBatch:
        value = batch["baseline_value"]
        bootstrap = batch.get("bootstrap")
        if bootstrap is None:
            bootstrap = ~batch["terminated"]
        bootstrap = bootstrap.to(value.dtype)
        continues = (~(batch["terminated"] | batch["truncated"])).to(value.dtype)
        delta = (
            batch[self.reward_field]
            + self.gamma * batch["baseline_next_value"] * bootstrap
            - value
        )
        advantage = discounted_suffix_sum(
            delta,
            continues.bool(),
            self.gamma * self.lambda_,
        )

        return batch.with_fields(
            advantage=advantage,
            returns=advantage + value,
        )


class DiscountedReturns:
    def __init__(self, gamma: float = 0.99, reward_field: str = "reward") -> None:
        self.gamma = gamma
        self.reward_field = reward_field

    @th.no_grad()
    def __call__(self, batch: TensorBatch, context: PrepareContext) -> TensorBatch:
        reward = batch[self.reward_field]
        continues = ~(batch["terminated"] | batch["truncated"])
        returns = discounted_suffix_sum(reward, continues, self.gamma)

        return batch.with_fields(returns=returns, advantage=returns)


class NStepTarget:
    def __init__(self, bootstrap, gamma: float = 0.99) -> None:
        self.bootstrap = bootstrap
        self.gamma = gamma

    @th.no_grad()
    def __call__(self, window: TensorBatch, context: PrepareContext) -> TensorBatch:
        if len(window.shape) < 2:
            raise ValueError("n-step targets require [time, batch, ...] windows")

        reward = window["reward"]
        target = th.zeros_like(reward[0])
        discount = 1.0

        for index in range(len(reward)):
            target += discount * reward[index]
            discount *= self.gamma

        bootstrap = window.get("bootstrap")
        if bootstrap is None:
            bootstrap = ~window["terminated"]
        target += (
            discount
            * bootstrap[-1].to(reward.dtype)
            * self.bootstrap(window["next_obs"][-1])
        )

        return window[0].with_fields(td_target=target)
