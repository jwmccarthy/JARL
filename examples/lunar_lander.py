from functools import partial

import gymnasium as gym
import torch.nn as nn

from jarl.collect import LogProbCapture, Runner, ValueCapture
from jarl.envs.wrappers import EpisodeStatsEnv
from jarl.modules.core import MLP
from jarl.modules.encoder.core import FlattenEncoder
from jarl.modules.operator import ValueFunction
from jarl.modules.policy import CategoricalPolicy
from jarl.modules.utils import init_layer
from jarl.store import RolloutBuffer


def make_environment():
    return EpisodeStatsEnv(gym.make("LunarLander-v3"))


def build_policy_and_value(environment, device: str):
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
    value_function = (
        ValueFunction(
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

    return policy, value_function


def build_collection(environment, policy, value_function, horizon: int, device: str):
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
            ValueCapture(value_function),
        ),
    )

    return runner, rollout
