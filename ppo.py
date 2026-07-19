import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

import gymnasium as gym

from jarl.envs.gym import SyncGymEnv

from jarl.collect import (
    LogProbCapture,
    PolicyVersionCapture,
    Runner,
    ValueCapture,
)
from jarl.learn import (
    OptimizerStep,
    PPOConfig,
    PPOLearner,
    PPOOptimizer,
    unique_parameters,
)
from jarl.sample import RolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.runtime import Evaluator, OnPolicySchedule, Trainer
from jarl.transform import GAE, SignRewards

from jarl.modules.core import MLP, CNN
from jarl.modules.utils import init_layer
from jarl.modules.operator import ValueFunction
from jarl.modules.policy import CategoricalPolicy
from jarl.modules.encoder.image import ImageEncoder

from jarl.envs.wrappers import EpisodeStatsEnv, ReshapeImageEnv

from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def make_env(env_id, **kwargs):
    def thunk():
        env = gym.make(env_id, **kwargs)
        env = EpisodeStatsEnv(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        # env = FireResetEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env = ReshapeImageEnv(env)
        return env

    return thunk

env = SyncGymEnv(make_env("ale_py:ALE/Pacman-v5", 
                       frameskip=1, 
                       repeat_action_probability=0.), 8)

policy = CategoricalPolicy(
    head=ImageEncoder(CNN(
        dims=[32, 64, 64],
        kernel=[8, 4, 3],
        stride=[4, 2, 1],
        init_func=lambda x: init_layer(x)
    ).append(nn.LazyLinear(512))),
    body=MLP(
        func=nn.ReLU, 
        dims=[], 
        init_func=lambda x: init_layer(x, std=0.01)
    ),
).build(env).to("cuda")

critic = ValueFunction(
    head=policy.head,    
    body=MLP(
        func=nn.ReLU, 
        dims=[],
        init_func=lambda x: init_layer(x, std=1)
    )
).build(env).to("cuda")

buffer = RolloutBuffer(128, env.n_envs, device="cuda")
runner = Runner(
    env,
    policy,
    buffer,
    captures=(
        LogProbCapture(),
        PolicyVersionCapture(policy),
        ValueCapture(critic),
    ),
)
training_vector_steps = int(1.25e6)
parameters = unique_parameters((policy, critic))
optimizer = Adam(parameters, lr=2.5e-4)
ppo = PPOLearner(
    transforms=(SignRewards(), GAE()),
    optimizer=PPOOptimizer(
        policy,
        critic,
        RolloutMinibatches(batch_size=256, epochs=4),
        OptimizerStep(
            (policy, critic),
            optimizer,
            max_grad_norm=0.5,
            scheduler=LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=(
                    training_vector_steps + buffer.horizon - 1
                )
                // buffer.horizon,
            ),
        ),
        PPOConfig(clip=0.1),
    ),
)

eval_env = SyncGymEnv(make_env("ale_py:ALE/Pacman-v5", 
                               frameskip=1, 
                               repeat_action_probability=0.), 1)

trainer = Trainer(
    runner,
    buffer,
    ppo,
    OnPolicySchedule(),
    checkpoint=Evaluator(eval_env, policy, path="checkpoints/pacman/ppo"),
)
trainer.run(training_vector_steps * env.n_envs)
