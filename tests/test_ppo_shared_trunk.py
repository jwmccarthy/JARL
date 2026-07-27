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


if __name__ == "__main__":
    unittest.main()
