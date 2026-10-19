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

    result = th.empty_like(value)
    running = th.zeros_like(value[0])

    for index in range(len(value) - 1, -1, -1):
        running = value[index] + discount * running * continues[index]
        result[index] = running

    return result


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


class SemiMarkovGAE:

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
        next_value = batch["baseline_next_value"]
        reward = batch[self.reward_field]
        duration = batch["duration"]
        valid = batch["valid"].bool()

        bootstrap = batch.get("bootstrap")
        if bootstrap is None:
            bootstrap = ~batch["terminated"]
        bootstrap = bootstrap.to(value.dtype)

        delta = (
            reward
            + (self.gamma ** duration) * next_value * bootstrap
            - value
        )

        done = batch["terminated"] | batch["truncated"]
        valid_float = valid.to(value.dtype)
        advantage = th.zeros_like(value)

        for step in reversed(range(len(value))):
            if step == len(value) - 1:
                continue_mask = th.zeros_like(valid[step])
                next_advantage = th.zeros_like(value[step])
            else:
                continue_mask = valid[step] & valid[step + 1] & ~done[step]
                next_advantage = advantage[step + 1]

            discount = (self.gamma ** duration[step]) * self.lambda_
            advantage[step] = valid_float[step] * (
                delta[step]
                + discount * continue_mask.to(value.dtype) * next_advantage
            )

        return batch.with_fields(
            advantage=advantage,
            returns=(advantage + value) * valid_float,
        )


class DiscountedReturns:

    def __init__(
        self,
        gamma:        float = 0.99,
        reward_field: str = "reward"
    ) -> None:
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
    def __call__(
        self,
        window:  TensorBatch,
        context: PrepareContext
    ) -> TensorBatch:
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
