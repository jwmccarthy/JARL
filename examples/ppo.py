import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

from jarl.envs.gym import SyncGymEnv
from jarl.learn import (
    Algorithm,
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    Update,
    unique_parameters,
)
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.sample import RolloutMinibatches
from jarl.transform import GAE

from lunar_lander import build_collection, build_policy_and_value, make_environment


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander-v3")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--checkpoint", type=Path)

    return parser.parse_args()


def build_ppo(environment, policy, value_function, rollout, arguments) -> Algorithm:
    policy_parameters = unique_parameters((policy, value_function))
    policy_optimizer = Adam(policy_parameters, lr=3e-4)
    vector_steps = arguments.total_timesteps // environment.n_envs
    update_count = (vector_steps + rollout.horizon - 1) // rollout.horizon
    ppo_update = Update(
        transforms=(GAE(gamma=0.99, lambda_=0.95),),
        sampler=RolloutMinibatches(
            batch_size=min(
                256,
                arguments.rollout_steps * arguments.num_envs,
            ),
            epochs=4,
        ),
        loss=PPOLoss(
            policy,
            value_function,
            PPOConfig(clip=0.2, entropy_coef=0.01),
        ),
        optimizer_step=OptimizerStep(
            (policy, value_function),
            policy_optimizer,
            max_grad_norm=0.5,
            scheduler=LinearLR(
                policy_optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=update_count,
            ),
        ),
        section="PPO",
    )

    return Algorithm(ppo_update)


def main() -> None:
    arguments = parse_arguments()

    if arguments.total_timesteps < arguments.num_envs:
        raise ValueError("total-timesteps must include at least one vector step")
    if arguments.rollout_steps < 1 or arguments.num_envs < 1:
        raise ValueError("num-envs and rollout-steps must be positive")

    device = arguments.device or ("cuda" if torch.cuda.is_available() else "cpu")
    environment = SyncGymEnv(make_environment, arguments.num_envs)
    policy, value_function = build_policy_and_value(environment, device)
    runner, rollout = build_collection(
        environment,
        policy,
        value_function,
        arguments.rollout_steps,
        device,
    )
    ppo = build_ppo(environment, policy, value_function, rollout, arguments)

    trainer = Trainer(runner, rollout, ppo, OnPolicySchedule())
    trainer.run(arguments.total_timesteps)

    if arguments.checkpoint:
        arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), arguments.checkpoint)


if __name__ == "__main__":
    main()
