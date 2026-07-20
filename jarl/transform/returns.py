import torch as th

from jarl.data.batch import TensorBatch
from jarl.transform.base import PrepareContext


class GAE:
    def __init__(
        self,
        gamma: float = 0.99,
        lambda_: float = 0.95,
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
        advantage = th.zeros_like(delta)
        carry = th.zeros_like(delta[-1])

        for index in reversed(range(len(delta))):
            carry = delta[index] + (
                self.gamma * self.lambda_ * continues[index] * carry
            )
            advantage[index] = carry

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
        returns = th.zeros_like(reward)
        carry = th.zeros_like(reward[-1])

        for index in reversed(range(len(reward))):
            carry = reward[index] + self.gamma * continues[index] * carry
            returns[index] = carry

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
