import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.envs.box2d.lunar_lander import heuristic
from torch.optim import Adam

from jarl.collect import LogProbCapture, PolicyVersionCapture, Runner, ValueCapture
from jarl.data import TensorBatch
from jarl.envs.gym import SyncGymEnv
from jarl.envs.wrappers import EpisodeStatsEnv
from jarl.learn import (
    GAIFOLoss,
    GAIFOMinibatches,
    LearningProgram,
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    RunUpdate,
    TransformRollout,
    Update,
    unique_parameters,
)
from jarl.modules.core import MLP
from jarl.modules.encoder.core import FlattenEncoder
from jarl.modules.operator import ValueFunction
from jarl.modules.policy import CategoricalPolicy
from jarl.modules.utils import init_layer
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.sample import RolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import DiscriminatorReward, GAE


class ExpertBuffer:
    def __init__(self, steps: int) -> None:
        env = gym.make("LunarLander-v3")
        observation, _ = env.reset(seed=0)
        observations = []
        next_observations = []

        for _ in range(steps):
            next_observation, _, terminated, truncated, _ = env.step(
                heuristic(env, observation)
            )
            observations.append(observation)
            next_observations.append(next_observation)
            observation = (
                env.reset()[0] if terminated or truncated else next_observation
            )

        env.close()
        self.transitions = TensorBatch(
            {
                "observation": torch.as_tensor(
                    np.asarray(observations), dtype=torch.float32
                ),
                "next_obs": torch.as_tensor(
                    np.asarray(next_observations), dtype=torch.float32
                ),
            }
        )

    def sample(self, batch_size: int) -> TensorBatch:
        indices = torch.randint(len(self.transitions), (batch_size,))
        return self.transitions[indices]


class TransitionDiscriminator(nn.Module):
    def __init__(self, observation_size: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, transition) -> torch.Tensor:
        observation, next_observation = transition
        inputs = torch.cat((observation, next_observation), dim=-1)
        return self.model(inputs).squeeze(-1)


def make_env():
    return EpisodeStatsEnv(gym.make("LunarLander-v3"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GAIfO on LunarLander-v3")
    parser.add_argument("--total-env-steps", type=int, default=500_000)
    parser.add_argument("--expert-steps", type=int, default=10_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_env_steps < args.num_envs:
        raise ValueError("total-env-steps must include at least one vector step")
    if min(args.expert_steps, args.num_envs, args.rollout_steps) < 1:
        raise ValueError("step and environment counts must be positive")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = SyncGymEnv(make_env, args.num_envs)
    expert_buffer = ExpertBuffer(args.expert_steps)

    policy = (
        CategoricalPolicy(
            head=FlattenEncoder(),
            body=MLP(
                dims=[64, 64],
                func=nn.Tanh,
                out_init_func=lambda layer: init_layer(layer, std=0.01),
            ),
        )
        .build(env)
        .to(device)
    )
    value_function = (
        ValueFunction(
            head=FlattenEncoder(),
            body=MLP(
                dims=[64, 64],
                func=nn.Tanh,
                out_init_func=lambda layer: init_layer(layer, std=1.0),
            ),
        )
        .build(env)
        .to(device)
    )
    discriminator = TransitionDiscriminator(env.obs_space.flat_dim).to(device)

    rollout = RolloutBuffer(args.rollout_steps, env.n_envs, device)
    runner = Runner(
        env,
        policy,
        rollout,
        captures=(
            LogProbCapture(),
            PolicyVersionCapture(policy),
            ValueCapture(value_function),
        ),
    )

    batch_size = min(256, args.rollout_steps * args.num_envs)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=3e-4)
    discriminator_update = Update(
        transforms=(),
        sampler=GAIFOMinibatches(
            expert_buffer,
            batch_size=batch_size,
            epochs=2,
        ),
        loss=GAIFOLoss(discriminator),
        optimizer_step=OptimizerStep(discriminator, discriminator_optimizer),
        section="Discriminator",
    )

    policy_parameters = unique_parameters((policy, value_function))
    policy_optimizer = Adam(policy_parameters, lr=3e-4)
    ppo_update = Update(
        transforms=(GAE(reward_field="imitation_reward"),),
        sampler=RolloutMinibatches(batch_size=batch_size, epochs=4),
        loss=PPOLoss(policy, value_function, PPOConfig()),
        optimizer_step=OptimizerStep(
            (policy, value_function),
            policy_optimizer,
            max_grad_norm=0.5,
        ),
        section="PPO",
    )

    gaifo = LearningProgram(
        (
            RunUpdate("experience", discriminator_update),
            TransformRollout(
                rollout="experience",
                output="rewarded",
                transforms=(DiscriminatorReward(discriminator),),
            ),
            RunUpdate("rewarded", ppo_update),
        )
    )

    trainer = Trainer(runner, rollout, gaifo, OnPolicySchedule())
    trainer.run(args.total_env_steps)

    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), args.checkpoint)


if __name__ == "__main__":
    main()
