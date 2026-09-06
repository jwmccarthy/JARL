import unittest

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn

from jarl.data.batch import TensorBatch
from jarl.envs.gym import SyncGymEnv
from jarl.learn.ppo import PPOConfig, PPOLoss
from jarl.modules.core import GRU, MLP
from jarl.modules.encoder.core import FlattenEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.sample.rollout import (
    ChunkBatch,
    RecurrentRolloutMinibatches,
    SequenceBatch,
    TrajectoryChunkMinibatches,
)
from jarl.transform import SemiMarkovGAE


class _MultiDiscreteEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, (4,), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete([2])

    def reset(self, seed=None, options=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}


class TestTrajectoryChunkMinibatches(unittest.TestCase):
    def _make_data(self, time: int, env: int):
        return TensorBatch(
            {
                "observation": th.arange(time * env * 3, dtype=th.float32).reshape(
                    time, env, 3
                ),
                "reward": th.arange(time * env, dtype=th.float32).reshape(time, env),
                "terminated": th.zeros(time, env, dtype=th.bool),
                "truncated": th.zeros(time, env, dtype=th.bool),
            }
        )

    def test_shapes_valid_and_duration(self):
        data = self._make_data(20, 3)
        data["terminated"][9, 0] = True
        data["terminated"][19, 0] = True
        data["truncated"][19, 1] = True
        data["terminated"][19, 2] = True

        sampler = TrajectoryChunkMinibatches(
            horizon=4, jitter=1, batch_size=12, epochs=1
        )
        batches = list(sampler(data))

        self.assertTrue(batches)
        for batch in batches:
            self.assertIsInstance(batch, ChunkBatch)
            max_duration = sampler.max_duration
            self.assertEqual(batch.data.shape[:2], (len(batch.duration), max_duration))
            self.assertEqual(batch.valid.shape, (len(batch.duration), max_duration))
            self.assertEqual(batch.duration.shape, (len(batch.duration),))
            self.assertEqual(batch.planned_duration.shape, batch.duration.shape)
            self.assertTrue((batch.duration >= 1).all())
            self.assertTrue((batch.duration <= max_duration).all())
            self.assertTrue((batch.duration <= batch.planned_duration).all())
            self.assertTrue(
                (batch.valid.sum(dim=1) == batch.duration).all()
            )
            self.assertLessEqual(
                batch.duration.sum().item(), sampler.batch_size
            )

    def test_valid_steps_per_batch_capped(self):
        data = self._make_data(10, 2)
        sampler = TrajectoryChunkMinibatches(
            horizon=3, jitter=1, batch_size=6, epochs=1
        )
        for batch in sampler(data):
            self.assertLessEqual(batch.duration.sum().item(), sampler.batch_size)

    def test_no_crossing_done_boundaries(self):
        data = self._make_data(10, 1)
        data["terminated"][4, 0] = True
        data["terminated"][9, 0] = True

        sampler = TrajectoryChunkMinibatches(
            horizon=3, jitter=0, batch_size=10, epochs=1
        )
        for batch in sampler(data):
            for env, start, end, _ in self._chunk_indices(batch, data):
                # No done inside a chunk except possibly at its last step.
                done_inside = data["terminated"][start : end - 1, env]
                self.assertFalse(done_inside.any().item())

    def test_counts_match_primitive_steps(self):
        data = self._make_data(12, 2)
        data["terminated"][5, 0] = True
        data["terminated"][11, 0] = True
        data["truncated"][11, 1] = True

        sampler = TrajectoryChunkMinibatches(
            horizon=4, jitter=2, batch_size=50, epochs=1
        )
        total_valid = sum(batch.duration.sum().item() for batch in sampler(data))
        self.assertEqual(total_valid, 12 * 2)

    def test_epochs_callback(self):
        data = self._make_data(8, 1)
        sampler = TrajectoryChunkMinibatches(
            horizon=3, jitter=0, batch_size=10, epochs=2
        )
        calls = []
        sampler.set_epoch_callback(lambda: calls.append(1))
        batches = list(sampler(data))
        self.assertEqual(len(calls), 2)
        self.assertTrue(batches)

    @staticmethod
    def _chunk_indices(batch: ChunkBatch, data: TensorBatch):
        # Reconstruct chunk indices from the preserved observation values.
        flat_data = data["observation"].flatten(0, 1)
        for index in range(len(batch.duration)):
            duration = int(batch.duration[index].item())
            valid_values = batch.data["observation"][index, :duration]
            match = (flat_data == valid_values[0]).all(dim=1)
            position = int(match.nonzero(as_tuple=True)[0][0].item())
            env = position // data.shape[0]
            start = position % data.shape[0]
            yield env, start, start + duration, duration

    def test_batch_must_fit_longest_planned_chunk(self):
        with self.assertRaises(ValueError):
            TrajectoryChunkMinibatches(horizon=4, jitter=2, batch_size=5)


class TestSemiMarkovGAE(unittest.TestCase):
    def _make_batch(self, *, decision_time, env, duration_value):
        value = th.zeros(decision_time, env)
        next_value = th.zeros(decision_time, env)
        reward = th.zeros(decision_time, env)
        duration = th.ones(decision_time, env, dtype=th.int64) * duration_value
        valid = th.ones(decision_time, env, dtype=th.bool)
        terminated = th.zeros(decision_time, env, dtype=th.bool)
        truncated = th.zeros(decision_time, env, dtype=th.bool)
        return TensorBatch(
            {
                "baseline_value": value,
                "baseline_next_value": next_value,
                "reward": reward,
                "duration": duration,
                "valid": valid,
                "terminated": terminated,
                "truncated": truncated,
            }
        )

    def test_invalid_entries_zero(self):
        batch = self._make_batch(decision_time=4, env=2, duration_value=2)
        batch["valid"][-1, :] = False
        batch["reward"][0, 0] = 1.0

        result = SemiMarkovGAE(gamma=0.9, lambda_=0.5)(batch, None)

        self.assertEqual(result["advantage"].shape, (4, 2))
        self.assertEqual(result["returns"].shape, (4, 2))
        self.assertTrue((result["advantage"][-1] == 0).all())
        self.assertTrue((result["returns"][-1] == 0).all())

    def test_no_gap_bridge(self):
        batch = self._make_batch(decision_time=5, env=1, duration_value=1)
        batch["reward"][0, 0] = 1.0
        batch["reward"][1, 0] = 1.0
        batch["reward"][4, 0] = 1.0
        batch["valid"][2, 0] = False
        batch["valid"][3, 0] = False

        result = SemiMarkovGAE(gamma=0.9, lambda_=0.5)(batch, None)

        # Without gap bridging advantage[0] would also include reward[4]. With
        # the gap it should only see reward[0] and reward[1].
        self.assertAlmostEqual(
            result["advantage"][0, 0].item(), 1.45, places=2
        )

    def test_manual_two_step(self):
        batch = self._make_batch(decision_time=3, env=1, duration_value=2)
        batch["reward"][0, 0] = 1.0
        batch["reward"][1, 0] = 1.0
        batch["valid"][-1, 0] = False

        gamma = 0.9
        lambda_ = 0.5
        result = SemiMarkovGAE(gamma=gamma, lambda_=lambda_)(batch, None)

        discount = (gamma ** 2) * lambda_  # 0.405
        expected_2 = 0.0  # invalid step
        expected_1 = 1.0 + discount * expected_2  # 1.0
        expected_0 = 1.0 + discount * expected_1  # 1.405

        self.assertAlmostEqual(
            result["advantage"][0, 0].item(), expected_0, places=6
        )
        self.assertAlmostEqual(
            result["advantage"][1, 0].item(), expected_1, places=6
        )
        self.assertAlmostEqual(
            result["advantage"][2, 0].item(), expected_2, places=6
        )


class TestRecurrentRolloutMinibatches(unittest.TestCase):
    def _make_data(self, time: int, env: int, seq_len: int, **extra):
        data = {
            "observation": th.zeros(time, env, 3),
            "policy_state": th.zeros(time, env, 5),
            "critic_state": th.zeros(time, env, 5),
            "terminated": th.zeros(time, env, dtype=th.bool),
            "truncated": th.zeros(time, env, dtype=th.bool),
            "learner_mask": th.ones(time, env, dtype=th.bool),
        }
        data.update(extra)
        return TensorBatch(data)

    def test_excludes_fully_invalid_sequences(self):
        time, env, seq_len = 8, 2, 4
        valid = th.ones(time, env, dtype=th.bool)
        valid[4:8, 1] = False
        data = self._make_data(time, env, seq_len, valid=valid)

        sampler = RecurrentRolloutMinibatches(seq_len, 2, epochs=1)
        for batch in sampler(data):
            self.assertTrue(batch.valid.any(dim=0).all().item())

    def test_combines_valid_and_learner_mask(self):
        time, env, seq_len = 4, 1, 2
        learner_mask = th.ones(time, env, dtype=th.bool)
        learner_mask[2, 0] = False
        valid = th.ones(time, env, dtype=th.bool)
        valid[3, 0] = False
        data = self._make_data(
            time, env, seq_len, learner_mask=learner_mask, valid=valid
        )

        sampler = RecurrentRolloutMinibatches(seq_len, 1, epochs=1)
        batches = list(sampler(data))
        # Sequence 1 (steps 2-3) is fully invalid and should be excluded.
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].valid.shape[1], 1)
        self.assertTrue(batches[0].valid[:, 0].all().item())

    def test_reset_ignores_invalid_padding(self):
        time, env, seq_len = 8, 1, 4
        valid = th.ones(time, env, dtype=th.bool)
        valid[6:8, 0] = False  # invalid padding at the end of the rollout
        terminated = th.zeros(time, env, dtype=th.bool)
        terminated[6, 0] = True  # done inside invalid padding
        data = self._make_data(
            time, env, seq_len, valid=valid, terminated=terminated
        )

        sampler = RecurrentRolloutMinibatches(seq_len, 1, epochs=1)
        batches = list(sampler(data))
        # The final sequence contains a done inside invalid padding, so no reset
        # mask should be produced.
        for batch in batches:
            if batch.valid.sum() < seq_len:
                self.assertIsNone(batch.reset)

    def test_historical_done_resets_before_learner_transition(self):
        time, env, seq_len = 4, 1, 4
        learner_mask = th.tensor([[False], [False], [True], [True]])
        terminated = th.tensor([[False], [True], [False], [False]])
        data = self._make_data(
            time,
            env,
            seq_len,
            learner_mask=learner_mask,
            terminated=terminated,
        )

        batch = next(iter(RecurrentRolloutMinibatches(seq_len, 1)(data)))

        self.assertIsNotNone(batch.reset)
        self.assertTrue(batch.reset[2, 0])
        self.assertFalse(batch.valid[:2, 0].any())
        self.assertTrue(batch.valid[2:, 0].all())


class TestPPOLossRecurrentCritic(unittest.TestCase):
    def _make_env(self, n_envs: int = 1):
        return SyncGymEnv(lambda: _MultiDiscreteEnv(), n_envs)

    def _make_sample(self, policy, critic, seq_len: int, batch: int):
        observation = th.zeros(seq_len, batch, 4)
        action = th.zeros(seq_len, batch, 1, dtype=th.int64)
        return SequenceBatch(
            steps=TensorBatch(
                {
                    "observation": observation,
                    "action": action,
                    "advantage": th.zeros(seq_len, batch),
                    "old_log_prob": th.zeros(seq_len, batch),
                    "returns": th.zeros(seq_len, batch),
                    "baseline_value": th.zeros(seq_len, batch),
                }
            ),
            initial_state=policy.initial_state(batch),
            reset=None,
            valid=th.ones(seq_len, batch, dtype=th.bool),
            initial_critic_state=critic.initial_state(batch)
            if critic.initial_state(batch) is not None
            else None,
        )

    def test_feedforward_critic_with_recurrent_policy_runs(self):
        env = self._make_env()
        policy = MultiCategoricalPolicy(
            foot=FlattenEncoder(),
            body=GRU(hidden_size=4, num_layers=1),
        ).build(env).to("cpu")
        critic = Critic(
            foot=FlattenEncoder(),
            body=MLP(dims=[4], func=nn.Tanh),
        ).build(env).to("cpu")

        loss_fn = PPOLoss(policy, critic, PPOConfig())
        sample = self._make_sample(policy, critic, seq_len=4, batch=1)
        output = loss_fn(sample)

        self.assertIsNotNone(output.loss)
        self.assertTrue(output.loss.requires_grad)

    def test_recurrent_critic_runs(self):
        env = self._make_env()
        policy = MultiCategoricalPolicy(
            foot=FlattenEncoder(),
            body=GRU(hidden_size=4, num_layers=1),
        ).build(env).to("cpu")
        critic = Critic(
            foot=FlattenEncoder(),
            body=GRU(hidden_size=4, num_layers=1),
        ).build(env).to("cpu")

        loss_fn = PPOLoss(policy, critic, PPOConfig())
        sample = self._make_sample(policy, critic, seq_len=4, batch=1)
        output = loss_fn(sample)

        self.assertIsNotNone(output.loss)
        self.assertTrue(output.loss.requires_grad)


if __name__ == "__main__":
    unittest.main()
