import unittest

import torch

from jarl.data import TensorBatch
from jarl.data.records import Evaluation
from jarl.learn import PPOConfig, PPOLoss


class SharedPolicy:
    def __init__(self, head, body, trunk):
        self.head = head
        self.body = body
        self.trunk = trunk
        self.feature_calls = 0

    def body_features(self, observation, state, reset):
        self.feature_calls += 1
        return self.trunk(observation), None

    def evaluate_from_features(self, features, observation, action):
        return Evaluation(log_prob=features[:, 0], entropy=features[:, 1])

    def evaluate_actions(self, *args, **kwargs):
        raise AssertionError("shared evaluation should use the shared features")


class SharedCritic:
    def __init__(self, head, body, value_head):
        self.head = head
        self.body = body
        self.value_head = value_head

    def value_from_features(self, features):
        return self.value_head(features).squeeze(-1)

    def evaluate_values(self, *args, **kwargs):
        raise AssertionError("shared evaluation should use the shared features")


class FactorPolicy:
    head = object()
    body = object()
    action_shape = (2,)
    sizes = (2, 3)

    def evaluate_actions(self, observation, action, state, reset=None):
        factor_entropy = torch.tensor(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        )
        return Evaluation(
            log_prob=torch.zeros(4),
            entropy=factor_entropy.sum(dim=-1),
            extras={"factor_entropy": factor_entropy},
        )


class FactorCritic:
    head = object()
    body = object()

    def evaluate_values(self, observation, state, reset=None):
        return torch.zeros(4)


class SharedTrunkPPOLossTests(unittest.TestCase):
    def test_shared_trunk_is_evaluated_once(self):
        trunk = torch.nn.Linear(3, 2)
        head = object()
        body = object()
        policy = SharedPolicy(head, body, trunk)
        critic = SharedCritic(head, body, torch.nn.Linear(2, 1))
        batch = TensorBatch(
            {
                "observation": torch.randn(4, 3),
                "action": torch.zeros(4, dtype=torch.int64),
                "advantage": torch.randn(4),
                "old_log_prob": torch.zeros(4),
                "baseline_value": torch.zeros(4),
                "returns": torch.randn(4),
            }
        )

        output = PPOLoss(policy, critic, PPOConfig())(batch)
        output.loss.backward()

        self.assertEqual(policy.feature_calls, 1)
        self.assertIsNotNone(trunk.weight.grad)

    def test_factor_entropy_and_action_rates_are_reported(self):
        batch = TensorBatch(
            {
                "observation": torch.zeros(4, 1),
                "action": torch.tensor([[0, 0], [1, 1], [1, 2], [0, 2]]),
                "advantage": torch.arange(4, dtype=torch.float32),
                "old_log_prob": torch.zeros(4),
                "baseline_value": torch.zeros(4),
                "returns": torch.zeros(4),
            }
        )

        output = PPOLoss(
            FactorPolicy(),
            FactorCritic(),
            PPOConfig(action_names=("first", "second")),
        )(batch)

        self.assertAlmostEqual(output.metrics["entropy_first"].item(), 0.4)
        self.assertAlmostEqual(output.metrics["entropy_second"].item(), 0.5)
        self.assertAlmostEqual(output.metrics["action_first_1_rate"].item(), 0.5)
        self.assertAlmostEqual(output.metrics["action_second_2_rate"].item(), 0.5)


if __name__ == "__main__":
    unittest.main()
