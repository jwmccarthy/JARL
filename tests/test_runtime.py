import unittest

import torch as th
import torch.nn as nn

from jarl.data import EnvStep, TensorBatch
from jarl.learn import (
    LearningProgram,
    OptimizerStep,
    PPOConfig,
    PPOLearner,
    PPOOptimizer,
    TrainDiscriminator,
    TransformArtifact,
    unique_parameters,
)
from jarl.modules.operator import ValueFunction
from jarl.modules.policy import CategoricalPolicy
from jarl.sample import RecurrentRolloutMinibatches, RolloutMinibatches
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.store import ReplayBuffer, Rollout, RolloutBuffer
from jarl.transform import DiscriminatorReward, GAE, NStepTarget


def transition(value: float, num_envs: int = 2):
    scalar = th.full((num_envs,), value)
    return {
        "obs": scalar[:, None],
        "act": th.zeros(num_envs, dtype=th.long),
        "rew": scalar,
        "next_obs": (scalar + 1)[:, None],
        "terminated": th.zeros(num_envs, dtype=th.bool),
        "truncated": th.zeros(num_envs, dtype=th.bool),
    }


class RuntimeTests(unittest.TestCase):
    def test_rollout_is_ordered_consumable_storage(self):
        buffer = RolloutBuffer(horizon=3, num_envs=2, device="cpu")
        for value in range(3):
            buffer.append(transition(float(value)))

        self.assertTrue(buffer.full)
        rollout = buffer.finish()
        th.testing.assert_close(
            rollout.steps["rew"][:, 0],
            th.tensor([0.0, 1.0, 2.0]),
        )
        with self.assertRaisesRegex(RuntimeError, "rollout is full"):
            buffer.append(transition(3.0))

        buffer.clear()
        self.assertEqual(buffer.position, 0)
        buffer.append(transition(4.0))
        th.testing.assert_close(buffer.finish().steps["rew"], th.full((1, 2), 4.0))

    def test_replay_samples_episode_safe_windows_after_wrap(self):
        replay = ReplayBuffer(capacity=3, num_envs=2)
        for value in range(4):
            replay.append(transition(float(value)))

        windows = replay.sample_windows(batch_size=4, length=2)
        self.assertEqual(windows.shape[:2], (2, 4))
        th.testing.assert_close(
            windows["rew"][1] - windows["rew"][0],
            th.ones(4),
        )

    def test_gae_handles_termination_and_time_order(self):
        batch = TensorBatch(
            {
                "rew": th.tensor([[1.0], [1.0]]),
                "terminated": th.tensor([[False], [True]]),
                "truncated": th.zeros(2, 1, dtype=th.bool),
                "baseline_value": th.zeros(2, 1),
                "baseline_next_value": th.zeros(2, 1),
            }
        )
        prepared = GAE(gamma=1.0, lambda_=1.0)(batch, None)
        th.testing.assert_close(prepared["adv"], th.tensor([[2.0], [1.0]]))
        th.testing.assert_close(prepared["ret"], prepared["adv"])

    def test_n_step_target_bootstraps_truncation_but_not_termination(self):
        base = {
            "rew": th.ones(2, 2),
            "next_obs": th.zeros(2, 2, 1),
            "terminated": th.tensor([[False, False], [True, False]]),
            "truncated": th.tensor([[False, False], [False, True]]),
        }
        target = NStepTarget(lambda obs: th.full((len(obs),), 4.0), gamma=0.5)(
            TensorBatch(base),
            None,
        )
        th.testing.assert_close(target["td_target"], th.tensor([1.5, 2.5]))

    def test_flat_sampler_covers_partial_final_batch(self):
        batch = TensorBatch({"value": th.arange(10).reshape(5, 2)})
        samples = list(RolloutMinibatches(batch_size=4)(batch))
        self.assertEqual([len(sample) for sample in samples], [4, 4, 2])
        combined = th.cat([sample["value"] for sample in samples]).sort().values
        th.testing.assert_close(combined, th.arange(10))

    def test_recurrent_sampler_emits_initial_state_and_reset_mask(self):
        done = th.tensor([[False], [True], [False], [False]])
        batch = TensorBatch(
            {
                "obs": th.arange(4.0).reshape(4, 1, 1),
                "terminated": done,
                "truncated": th.zeros_like(done),
                "behavior_state": th.arange(4.0).reshape(4, 1, 1),
            }
        )
        sample = next(
            iter(
                RecurrentRolloutMinibatches(
                    sequence_length=4,
                    sequences_per_batch=1,
                )(batch)
            )
        )
        th.testing.assert_close(sample.initial_state, th.tensor([[0.0]]))
        th.testing.assert_close(
            sample.reset[:, 0],
            th.tensor([False, False, True, False]),
        )

    def test_learning_program_validates_and_preserves_stage_order(self):
        order = []

        class Stage:
            def __init__(self, required, produced, name):
                self.requires = frozenset(required)
                self.produces = frozenset(produced)
                self.name = name

            def run(self, workspace):
                order.append(self.name)
                for artifact in self.produces:
                    workspace.publish(artifact, artifact)

        program = LearningProgram(
            [
                Stage({"source"}, {"rewarded"}, "reward"),
                Stage({"rewarded"}, {"prepared"}, "prepare"),
                Stage({"prepared"}, set(), "optimize"),
            ]
        )
        program.update(object())
        self.assertEqual(order, ["reward", "prepare", "optimize"])

        with self.assertRaisesRegex(ValueError, "unavailable artifacts"):
            LearningProgram([Stage({"missing"}, set(), "invalid")])

    def test_gaifo_stage_trains_discriminator_before_reward_materialization(self):
        class Discriminator(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 1)

            def forward(self, pair):
                return self.linear(th.cat(pair, dim=-1)).squeeze(-1)

        class ExpertSource:
            def sample(self, batch_size):
                return TensorBatch(
                    {
                        "obs": th.zeros(batch_size, 1),
                        "next_obs": th.zeros(batch_size, 1),
                    }
                )

        captured_reward = []

        class CaptureReward:
            requires = frozenset({"rewarded"})
            produces = frozenset()

            def run(self, workspace):
                captured_reward.append(
                    workspace.require("rewarded").steps["imitation_rew"]
                )

        discriminator = Discriminator()
        optimizer = th.optim.Adam(discriminator.parameters(), lr=1e-2)
        rollout = Rollout(
            TensorBatch(
                {
                    "obs": th.ones(4, 1, 1),
                    "next_obs": th.ones(4, 1, 1) * 2,
                }
            )
        )
        program = LearningProgram(
            [
                TrainDiscriminator(
                    source="source",
                    expert_source=ExpertSource(),
                    discriminator=discriminator,
                    optimizer_step=OptimizerStep(discriminator, optimizer),
                    batch_size=2,
                ),
                TransformArtifact(
                    source="source",
                    output="rewarded",
                    transforms=(DiscriminatorReward(discriminator),),
                ),
                CaptureReward(),
            ]
        )

        metrics = program.update(rollout)

        self.assertIn("Discriminator", metrics)
        self.assertEqual(captured_reward[0].shape, (4, 1))

    def test_feed_forward_ppo_updates_and_advances_model_versions(self):
        policy = CategoricalPolicy(nn.Identity(), nn.Identity())
        policy.model = nn.Linear(2, 2)
        critic = ValueFunction(nn.Identity(), nn.Identity())
        critic.model = nn.Linear(2, 1)
        obs = th.randn(4, 2, 2)
        next_obs = th.randn(4, 2, 2)
        with th.no_grad():
            decision = policy.act(obs)
            value = critic.value(obs)
            next_value = critic.value(next_obs)
        rollout = Rollout(
            TensorBatch(
                {
                    "obs": obs,
                    "act": decision.action,
                    "rew": th.randn(4, 2),
                    "next_obs": next_obs,
                    "terminated": th.zeros(4, 2, dtype=th.bool),
                    "truncated": th.zeros(4, 2, dtype=th.bool),
                    "behavior_log_prob": decision.artifacts["log_prob"],
                    "policy_version": th.zeros(4, 2, dtype=th.long),
                    "baseline_value": value,
                    "baseline_next_value": next_value,
                    "value_version": th.zeros(4, 2, dtype=th.long),
                }
            )
        )
        optimizer = th.optim.Adam(unique_parameters((policy, critic)), lr=1e-3)
        learner = PPOLearner(
            transforms=(GAE(),),
            optimizer=PPOOptimizer(
                policy,
                critic,
                RolloutMinibatches(batch_size=4),
                OptimizerStep((policy, critic), optimizer),
                PPOConfig(),
            ),
        )

        metrics = learner.update(rollout)

        self.assertEqual(policy.version, 1)
        self.assertEqual(critic.version, 1)
        self.assertEqual(
            set(metrics["PPO"]),
            {"policy_loss", "critic_loss", "entropy", "approx_kl"},
        )
        with self.assertRaisesRegex(RuntimeError, "current policy version"):
            learner.update(rollout)

    def test_trainer_updates_full_and_partial_rollouts_without_overwrite(self):
        source = RolloutBuffer(horizon=3, num_envs=2, device="cpu")

        class Runner:
            n_envs = 2
            behavior = object()

            def __init__(self):
                self.index = 0

            def reset(self):
                return None

            def step(self):
                source.append(transition(float(self.index)))
                self.index += 1
                return EnvStep(
                    next_obs=None,
                    collector_obs=None,
                    reward=th.zeros(2),
                    terminated=th.zeros(2, dtype=th.bool),
                    truncated=th.zeros(2, dtype=th.bool),
                    info={},
                )

        class Learner:
            def __init__(self):
                self.rollouts = []

            def update(self, rollout):
                self.rollouts.append(rollout.steps["rew"].clone())
                return {"Test": {"updates": 1.0}}

        class Logger:
            def progress(self, steps):
                return range(steps)

            def episode(self, step, info):
                return None

            def update(self, metrics, step=None):
                return None

        learner = Learner()
        trainer = Trainer(
            Runner(),
            source,
            learner,
            OnPolicySchedule(),
            logger=Logger(),
        )
        trainer.run(total_env_steps=10)

        self.assertEqual([len(rollout) for rollout in learner.rollouts], [3, 2])
        th.testing.assert_close(
            learner.rollouts[0][:, 0],
            th.tensor([0.0, 1.0, 2.0]),
        )
        th.testing.assert_close(
            learner.rollouts[1][:, 0],
            th.tensor([3.0, 4.0]),
        )


if __name__ == "__main__":
    unittest.main()
