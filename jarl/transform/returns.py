import torch as th

from jarl.data.batch import TensorBatch
from jarl.transform.base import PrepareContext


class GAE:
    produces = frozenset({"adv", "ret"})
    replaces = frozenset()

    def __init__(
        self,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        reward_field: str = "rew",
    ) -> None:
        self.gamma = gamma
        self.lambda_ = lambda_
        self.reward_field = reward_field
        self.requires = frozenset(
            {
                reward_field,
                "terminated",
                "truncated",
                "baseline_value",
                "baseline_next_value",
            }
        )

    @th.no_grad()
    def __call__(
        self, batch: TensorBatch, context: PrepareContext
    ) -> TensorBatch:
        value = batch["baseline_value"]
        bootstrap = (~batch["terminated"]).to(value.dtype)
        continuation = (~(batch["terminated"] | batch["truncated"])).to(
            value.dtype
        )
        delta = (
            batch[self.reward_field]
            + self.gamma * batch["baseline_next_value"] * bootstrap
            - value
        )
        adv = th.zeros_like(delta)
        carry = th.zeros_like(delta[-1])
        for index in reversed(range(len(delta))):
            carry = delta[index] + (
                self.gamma * self.lambda_ * continuation[index] * carry
            )
            adv[index] = carry
        return batch.with_fields(adv=adv, ret=adv + value)


class DiscountedReturns:
    produces = frozenset({"ret", "adv"})
    replaces = frozenset()

    def __init__(self, gamma: float = 0.99, reward_field: str = "rew") -> None:
        self.gamma = gamma
        self.reward_field = reward_field
        self.requires = frozenset({reward_field, "terminated", "truncated"})

    @th.no_grad()
    def __call__(
        self, batch: TensorBatch, context: PrepareContext
    ) -> TensorBatch:
        reward = batch[self.reward_field]
        continuation = ~(batch["terminated"] | batch["truncated"])
        returns = th.zeros_like(reward)
        carry = th.zeros_like(reward[-1])
        for index in reversed(range(len(reward))):
            carry = reward[index] + self.gamma * continuation[index] * carry
            returns[index] = carry
        return batch.with_fields(ret=returns, adv=returns)


class NStepTarget:
    requires = frozenset({"rew", "next_obs", "terminated", "truncated"})
    produces = frozenset({"td_target"})
    replaces = frozenset()

    def __init__(self, bootstrap, gamma: float = 0.99) -> None:
        self.bootstrap = bootstrap
        self.gamma = gamma

    @th.no_grad()
    def __call__(
        self, window: TensorBatch, context: PrepareContext
    ) -> TensorBatch:
        if len(window.shape) < 2:
            raise ValueError("n-step targets require [time, batch, ...] windows")
        reward = window["rew"]
        target = th.zeros_like(reward[0])
        discount = 1.0
        for index in range(len(reward)):
            target += discount * reward[index]
            discount *= self.gamma
        target += (
            discount
            * (~window["terminated"][-1]).to(reward.dtype)
            * self.bootstrap(window["next_obs"][-1])
        )
        return window[0].with_fields(td_target=target)
