import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn

from jarl.collect import (
    CaptureContext,
    LogProbCapture,
    RecurrentStateCapture,
    Runner,
    ValueCapture,
    build_record,
)
from jarl.data import EnvStep, PolicyOutput
from jarl.envs import SyncGymEnv
from jarl.modules.policy import CategoricalPolicy


class FakeValue:
    version = 0

    def value(self, observation, state=None):
        return observation.sum(dim=-1)


class FakePolicy:
    device = "cpu"

    def initial_state(self, batch_size):
        return th.zeros(batch_size, 2)

    def act(self, observation, state=None, *, deterministic=False):
        action = th.arange(len(observation))
        return PolicyOutput(
            action=action,
            next_state=state + 1,
            log_prob=-action.float(),
        )


class FakeEnv:
    n_envs = 2

    def reset(self):
        return np.array([[1.0], [2.0]], dtype=np.float32)

    def step(self, action):
        np.testing.assert_array_equal(action.numpy(), np.array([0, 1]))
        return EnvStep(
            next_obs=np.array([[3.0], [4.0]], dtype=np.float32),
            observation=np.array([[3.0], [9.0]], dtype=np.float32),
            reward=np.array([0.5, 1.0], dtype=np.float32),
            terminated=np.array([False, True]),
            truncated=np.array([False, False]),
            info={"reward": [1.0], "length": [4]},
        )


class FakeBuffer:
    def __init__(self):
        self.records = []

    def append(self, record):
        self.records.append(record)


class CollectTests(unittest.TestCase):
    def test_env_step_keeps_boundaries_distinct(self):
        step = EnvStep(None, None, None, th.tensor([False]), th.tensor([True]))
        self.assertTrue(step.done.item())
        self.assertFalse(step.terminated.item())
        self.assertTrue(step.truncated.item())

    def test_build_record_uses_explicit_captures(self):
        observation = th.tensor([[1.0], [2.0]], requires_grad=True)
        state = th.zeros(2, 2)
        policy_output = FakePolicy().act(observation, state)
        step = FakeEnv().step(policy_output.action)
        captures = (
            LogProbCapture(),
            ValueCapture(FakeValue()),
            RecurrentStateCapture(),
        )
        record = build_record(
            CaptureContext(observation, state, policy_output, step),
            captures,
        )

        self.assertEqual(
            set(record),
            {
                "observation",
                "action",
                "reward",
                "next_obs",
                "terminated",
                "truncated",
                "old_log_prob",
                "baseline_value",
                "baseline_next_value",
                "value_version",
                "policy_state",
            },
        )
        th.testing.assert_close(record["baseline_value"], th.tensor([1.0, 2.0]))
        th.testing.assert_close(
            record["baseline_next_value"], th.tensor([3.0, 4.0])
        )
        self.assertFalse(record["baseline_value"].requires_grad)

    def test_runner_stores_one_record_and_resets_done_state(self):
        buffer = FakeBuffer()
        runner = Runner(
            FakeEnv(),
            FakePolicy(),
            buffer,
            captures=(
                LogProbCapture(),
                RecurrentStateCapture(),
            ),
        )
        runner.reset()
        step = runner.step()

        self.assertEqual(step.info, {"reward": [1.0], "length": [4]})
        self.assertEqual(len(buffer.records), 1)
        np.testing.assert_array_equal(
            runner.observation,
            np.array([[3.0], [9.0]]),
        )
        th.testing.assert_close(runner.state, th.tensor([[1.0, 1.0], [0.0, 0.0]]))
        th.testing.assert_close(
            buffer.records[0]["policy_state"], th.zeros(2, 2)
        )

    def test_policy_output_and_evaluation_share_distribution_semantics(self):
        policy = CategoricalPolicy(nn.Identity(), nn.Identity())
        policy.model = nn.Linear(2, 3, bias=False)
        with th.no_grad():
            policy.model.weight.copy_(
                th.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
            )
        observation = th.tensor([[1.0, 2.0], [2.0, 1.0]])

        policy_output = policy.act(observation)
        evaluation = policy.evaluate_actions(observation, policy_output.action)

        th.testing.assert_close(policy_output.log_prob, evaluation.log_prob)
        deterministic = policy.act(observation, deterministic=True)
        th.testing.assert_close(deterministic.action, policy.action(observation))

    def test_sync_gym_preserves_transition_and_policy_observations(self):
        class TruncatingEnv:
            observation_space = gym.spaces.Box(-1, 10, shape=(1,), dtype=np.float32)
            action_space = gym.spaces.Discrete(2)

            def __init__(self):
                self.resets = 0

            def reset(self):
                self.resets += 1
                return np.array([9.0], dtype=np.float32), {}

            def step(self, action):
                return (
                    np.array([5.0], dtype=np.float32),
                    1.0,
                    False,
                    True,
                    SimpleNamespace(reward=3.0, length=4),
                )

        env = SyncGymEnv(TruncatingEnv, n_envs=1)
        env.reset()
        step = env.step(th.tensor([0]))

        np.testing.assert_array_equal(step.next_obs, np.array([[5.0]]))
        np.testing.assert_array_equal(step.observation, np.array([[9.0]]))
        np.testing.assert_array_equal(step.terminated, np.array([False]))
        np.testing.assert_array_equal(step.truncated, np.array([True]))
        self.assertEqual(step.info, {"reward": [3.0], "length": [4]})


if __name__ == "__main__":
    unittest.main()
