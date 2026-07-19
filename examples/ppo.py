import argparse
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

from jarl.collect import LogProbCapture, PolicyVersionCapture, Runner, ValueCapture
from jarl.envs.gym import SyncGymEnv
from jarl.envs.wrappers import EpisodeStatsEnv
from jarl.learn import OptimizerStep, PPOConfig, PPOLoss, Update, unique_parameters
from jarl.modules.core import MLP
from jarl.modules.encoder.core import FlattenEncoder
from jarl.modules.operator import ValueFunction
from jarl.modules.policy import CategoricalPolicy
from jarl.modules.utils import init_layer
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.sample import RolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE


def make_env():
    return EpisodeStatsEnv(gym.make("LunarLander-v3"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander-v3")
    parser.add_argument("--total-env-steps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_env_steps < args.num_envs:
        raise ValueError("total-env-steps must include at least one vector step")
    if args.rollout_steps < 1 or args.num_envs < 1:
        raise ValueError("num-envs and rollout-steps must be positive")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = SyncGymEnv(make_env, args.num_envs)

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

    rollout = RolloutBuffer(
        horizon=args.rollout_steps,
        num_envs=env.n_envs,
        device=device,
    )
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

    parameters = unique_parameters((policy, value_function))
    optimizer = Adam(parameters, lr=3e-4)
    vector_steps = args.total_env_steps // env.n_envs
    updates = (vector_steps + rollout.horizon - 1) // rollout.horizon
    ppo = Update(
        transforms=(GAE(gamma=0.99, lambda_=0.95),),
        sampler=RolloutMinibatches(
            batch_size=min(256, args.rollout_steps * args.num_envs),
            epochs=4,
        ),
        loss=PPOLoss(
            policy,
            value_function,
            PPOConfig(clip=0.2, entropy_coef=0.01),
        ),
        optimizer_step=OptimizerStep(
            (policy, value_function),
            optimizer,
            max_grad_norm=0.5,
            scheduler=LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=updates,
            ),
        ),
        section="PPO",
    )

    trainer = Trainer(runner, rollout, ppo, OnPolicySchedule())
    trainer.run(args.total_env_steps)

    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), args.checkpoint)


if __name__ == "__main__":
    main()
