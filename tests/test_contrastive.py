"""GPU-safe future goals, terminal transitions, and off-policy progress."""

import unittest
from contextlib import contextmanager

import torch as th
import torch.nn as nn

from jarl.collect import GoalConditionedRunner, ReplayGoalSampler
from jarl.learn import Algorithm, ContrastiveLearner, ContrastiveUpdate
from jarl.runtime import OffPolicySchedule, Trainer
from jarl.store import FutureGoalReplayBuffer


def joint_goal(observation: th.Tensor) -> th.Tensor:
    return th.cat((observation[:, :3], observation[:, 4:7]), dim=-1)


class SimpleCodec(nn.Module):
    def mask(self, observation: th.Tensor) -> th.Tensor:
        mask = th.ones(len(observation), 5, dtype=th.bool, device=observation.device)
        mask[:, 1] = False
        return mask


class FakeEnv:
    n_envs = 2

    def __init__(self, device: th.device) -> None:
        self.device = device

    def reset(self) -> th.Tensor:
        return th.zeros(2, 8, device=self.device)

    def step(self, action: th.Tensor):
        assert action.device == self.device and action.shape == (2, 2)
        observation = th.zeros(2, 8, device=self.device)
        observation[0, :3] = -9  # CARL autoreset, not the terminal achievement.
        observation[1, :3] = 1
        observation[1, 4:7] = 2
        final = th.zeros_like(observation)
        final[0, :3] = 1       # Ball succeeds, car does not.
        return (
            observation, th.tensor([1., 0.], device=self.device),
            th.tensor([True, False], device=self.device),
            th.zeros(2, dtype=th.bool, device=self.device),
            {"final_obs": final},
        )


class RecordingLogger:
    def __init__(self) -> None:
        self.activities = []
        self.metrics = []
        self.main_steps = 0

    @contextmanager
    def progress(self, total_steps: int, initial_steps: int = 0):
        self.main_steps = initial_steps
        yield

    def start_activity(self, total: int, description: str):
        self.activities.append(("start", description, total))

    def advance_activity(self, amount: int = 1):
        self.activities.append(("advance", amount))

    def finish_activity(self):
        self.activities.append(("finish",))

    def advance(self, amount: int = 1):
        self.main_steps += amount

    def update(self, metrics, step=None):
        self.metrics.append((metrics, step))


def make_learner(device: th.device) -> ContrastiveLearner:
    return ContrastiveLearner(
        8, 6, SimpleCodec().to(device), (2, 3),
        actor_width=8, actor_depth=4, critic_width=8, critic_depth=4,
        embedding_size=4, actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3,
        entropy_target_fraction=0.5, logsumexp_penalty=0.1, device=device,
    )


def make_replay(device: th.device) -> FutureGoalReplayBuffer:
    return FutureGoalReplayBuffer(
        4, 2, 8, 6, (2,), joint_goal, 0.5, 4, device,
    )


class ContrastiveComponentsTests(unittest.TestCase):
    def test_final_observation_and_joint_success_with_gpu_native_replay(self):
        for device in ([th.device("cpu"), th.device("cuda:0")]
                       if th.cuda.is_available() else [th.device("cpu")]):
            with self.subTest(device=device):
                replay = make_replay(device)
                learner = make_learner(device)
                logger = RecordingLogger()
                expert = th.tensor([[1., 1., 1., 2., 2., 2.]], device=device)
                sampler = ReplayGoalSampler(
                    replay, joint_goal, expert_goals=expert, expert_fraction=1,
                )
                runner = GoalConditionedRunner(
                    FakeEnv(device), learner.actor, replay, joint_goal, sampler,
                    goal_horizon=3, goal_tolerance=0.01,
                    components={"ball": slice(0, 3), "car": slice(3, 6)},
                    logger=logger, report_interval=1, prefill_steps=1, collect_steps=3,
                )
                runner.reset()
                self.assertEqual(runner.goals.device, device)
                step = runner.step()
                self.assertEqual(step.next_obs.device, device)
                self.assertEqual(replay.observations.device, device)
                self.assertEqual(replay.achieved.device, device)
                self.assertEqual(replay.current_episode.device, device)
                self.assertEqual(replay.future_weights.device, device)
                self.assertEqual(replay.sample_goals(4).device, device)
                batch = replay.sample(4)
                self.assertTrue(all(
                    tensor.device == device
                    for tensor in (batch.observation, batch.action,
                                   batch.goal, batch.steps_to_goal)
                ))
                th.testing.assert_close(replay.achieved[0, 0, :3], expert[0, :3])
                th.testing.assert_close(step.observation[0, :3],
                                        th.full((3,), -9., device=device))
                self.assertEqual(runner.last_metrics["ball_near_goal"], 1.0)
                self.assertEqual(runner.last_metrics["car_near_goal"], 0.5)
                self.assertEqual(runner.last_metrics["near_goal"], 0.5)
                self.assertEqual(replay.current_episode[0].item(), 1)
                self.assertIn(("start", "prefill replay", 1), logger.activities)
                self.assertIn(("finish",), logger.activities)

    def test_trainer_flushes_partial_collection_and_reports_each_update(self):
        device = th.device("cpu")
        replay = make_replay(device)
        learner = make_learner(device)
        logger = RecordingLogger()
        sampler = ReplayGoalSampler(replay, joint_goal)
        runner = GoalConditionedRunner(
            FakeEnv(device), learner.actor, replay, joint_goal, sampler,
            goal_horizon=2, goal_tolerance=0.1, logger=logger, report_interval=1,
            prefill_steps=1, collect_steps=3,
        )
        update = ContrastiveUpdate(learner, 4, 2)
        trainer = Trainer(
            runner, replay, Algorithm(update),
            OffPolicySchedule(2, update_every_vector_steps=3,
                              min_replay_vector_steps=1, flush_partial=True),
            logger=logger, track_episodes=False,
        )
        actor_before = learner.actor.network.head.weight.detach().clone()
        critic_before = learner.critic.state_action.head.weight.detach().clone()
        trainer.run(4)  # Two vector steps, shorter than the three-step block.
        self.assertEqual(trainer.clock.env_steps, 4)
        self.assertEqual(trainer.clock.episodes, 0)
        self.assertEqual(update.gradient_steps, 2)
        self.assertEqual(logger.main_steps, 4)
        self.assertIn(("start", "CRL optimizer", 2), logger.activities)
        self.assertEqual(sum(event[0] == "advance" for event in logger.activities), 4)
        self.assertTrue(any("CRL" in metrics for metrics, _ in logger.metrics))
        self.assertGreater((learner.actor.network.head.weight - actor_before).abs().sum(), 0)
        self.assertGreater((learner.critic.state_action.head.weight - critic_before).abs().sum(), 0)


if __name__ == "__main__":
    unittest.main()
