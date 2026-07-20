import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.envs.box2d.lunar_lander import heuristic
from torch.optim import Adam

from jarl.data import TensorBatch
from jarl.envs.gym import SyncGymEnv
from jarl.learn import (
    Algorithm,
    GAIFOLoss,
    GAIFOMinibatches,
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    TransformRollout,
    Update,
    unique_parameters,
)
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.sample import RolloutMinibatches
from jarl.transform import DiscriminatorReward, GAE

from lunar_lander import build_collection, build_policy_and_value, make_environment


class ExpertBuffer:
    def __init__(self, steps: int) -> None:
        self.transitions = self._collect(steps)

    @staticmethod
    def _collect(steps: int) -> TensorBatch:
        environment = gym.make("LunarLander-v3")
        observation, _ = environment.reset(seed=0)
        observations = []
        next_observations = []

        for _ in range(steps):
            next_observation, _, terminated, truncated, _ = environment.step(
                heuristic(environment, observation)
            )
            observations.append(observation)
            next_observations.append(next_observation)

            observation = (
                environment.reset()[0] if terminated or truncated else next_observation
            )

        environment.close()

        return TensorBatch(
            {
                "observation": torch.as_tensor(
                    np.asarray(observations),
                    dtype=torch.float32,
                ),
                "next_obs": torch.as_tensor(
                    np.asarray(next_observations),
                    dtype=torch.float32,
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
        transition = torch.cat((observation, next_observation), dim=-1)

        return self.model(transition).squeeze(-1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GAIfO on LunarLander-v3")
    parser.add_argument("--total-env-steps", type=int, default=500_000)
    parser.add_argument("--expert-steps", type=int, default=10_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--checkpoint", type=Path)

    return parser.parse_args()


def build_discriminator_update(
    discriminator,
    expert_buffer,
    batch_size: int,
) -> Update:
    discriminator_optimizer = Adam(discriminator.parameters(), lr=3e-4)

    return Update(
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


def build_ppo_update(policy, value_function, batch_size: int) -> Update:
    policy_parameters = unique_parameters((policy, value_function))
    policy_optimizer = Adam(policy_parameters, lr=3e-4)

    return Update(
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


def build_gaifo(
    policy,
    value_function,
    discriminator,
    expert_buffer,
    batch_size: int,
) -> Algorithm:
    discriminator_update = build_discriminator_update(
        discriminator,
        expert_buffer,
        batch_size,
    )
    ppo_update = build_ppo_update(policy, value_function, batch_size)

    return Algorithm(
        discriminator_update,
        TransformRollout(DiscriminatorReward(discriminator)),
        ppo_update,
    )


def main() -> None:
    arguments = parse_arguments()

    if arguments.total_env_steps < arguments.num_envs:
        raise ValueError("total-env-steps must include at least one vector step")
    if (
        min(
            arguments.expert_steps,
            arguments.num_envs,
            arguments.rollout_steps,
        )
        < 1
    ):
        raise ValueError("step and environment counts must be positive")

    device = arguments.device or ("cuda" if torch.cuda.is_available() else "cpu")
    environment = SyncGymEnv(make_environment, arguments.num_envs)
    expert_buffer = ExpertBuffer(arguments.expert_steps)
    policy, value_function = build_policy_and_value(environment, device)
    discriminator = TransitionDiscriminator(environment.obs_space.flat_dim).to(device)
    runner, rollout = build_collection(
        environment,
        policy,
        value_function,
        arguments.rollout_steps,
        device,
    )

    batch_size = min(256, arguments.rollout_steps * arguments.num_envs)
    gaifo = build_gaifo(
        policy,
        value_function,
        discriminator,
        expert_buffer,
        batch_size,
    )

    trainer = Trainer(runner, rollout, gaifo, OnPolicySchedule())
    trainer.run(arguments.total_env_steps)

    if arguments.checkpoint:
        arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), arguments.checkpoint)


if __name__ == "__main__":
    main()
