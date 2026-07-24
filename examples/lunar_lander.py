from functools import partial

import gymnasium as gym
import torch.nn as nn

from jarl.collect import CriticCapture, LogProbCapture, Runner
from jarl.envs.wrappers import EpisodeStatsEnv
from jarl.modules.core import MLP
from jarl.modules.encoder.core import FlattenEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import CategoricalPolicy
from jarl.modules.utils import init_layer
from jarl.store import RolloutBuffer


def make_environment():
    return EpisodeStatsEnv(gym.make("LunarLander-v3"))


def build_policy_and_critic(environment, device: str):
    policy = (
        CategoricalPolicy(
            head=FlattenEncoder(),
            body=MLP(
                dims=[64, 64],
                func=nn.Tanh,
                out_init_func=partial(init_layer, std=0.01),
            ),
        )
        .build(environment)
        .to(device)
    )
    critic = (
        Critic(
            head=FlattenEncoder(),
            body=MLP(
                dims=[64, 64],
                func=nn.Tanh,
                out_init_func=partial(init_layer, std=1.0),
            ),
        )
        .build(environment)
        .to(device)
    )

    return policy, critic


def build_collection(environment, policy, critic, horizon: int, device: str):
    rollout = RolloutBuffer(
        horizon=horizon,
        num_envs=environment.n_envs,
        device=device,
    )
    runner = Runner(
        environment,
        policy,
        rollout,
        captures=(
            LogProbCapture(),
            CriticCapture(critic),
        ),
    )

    return runner, rollout
