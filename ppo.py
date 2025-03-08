import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

import gymnasium as gym

from jarl.envs.env import SyncEnv

from jarl.data.buffer import LazyBuffer

from jarl.modules.core import MLP, CNN
from jarl.modules.operator import Critic
from jarl.modules.policy import CategoricalPolicy
from jarl.modules.encoder.image import ImageEncoder

from jarl.train.update.ppo import PPOUpdate
from jarl.train.sample.batch import BatchSampler
from jarl.train.optim import Optimizer, Scheduler

from jarl.train.loop import TrainLoop
from jarl.train.graph import TrainGraph

from jarl.train.modify.compute import (
    ComputeValues,
    ComputeLogProbs,
    ComputeAdvantages,
    ComputeReturns,
    SignRewards
)

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def make_env(env_id):
    def thunk():
        env = gym.make(env_id, frameskip=1)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        # env = EpisodicLifeEnv(env)
        env = FireResetEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env

    return thunk

env = SyncEnv(make_env("ale_py:ALE/Breakout-v5"), 8, device="cuda")

policy = CategoricalPolicy(
    head=ImageEncoder(CNN(
        dims=[32, 64, 64],
        kernel=[8, 4, 3],
        stride=[4, 2, 1]
    ).append(nn.LazyLinear(512))),
    body=MLP(func=nn.ReLU, dims=[])
).build(env).to("cuda")

critic = Critic(
    head=policy.head,    
    body=MLP(func=nn.ReLU, dims=[]),
).build(env).to("cuda")

ppo = (
    TrainGraph(
        PPOUpdate(128, policy, critic, clip=0.1,
                  optimizer=Optimizer(Adam, lr=2.5e-4),
                  scheduler=Scheduler(LinearLR, start_factor=1.0, end_factor=0.0)),
        BatchSampler(256, num_epoch=4)
    )
    .add_modifier(ComputeAdvantages())
    .add_modifier(ComputeLogProbs(policy))
    .add_modifier(ComputeReturns())
    .add_modifier(ComputeValues(critic))
    .add_modifier(SignRewards())
    .compile()
)

buffer = LazyBuffer(128).to("cuda")

loop = TrainLoop(env, buffer, policy, graphs=[ppo])
loop.run(int(1.25e6))


# env = SyncEnv(make_env("ALE/Breakout-v5", render=True), 1, device="cuda")

# N = 65536

# obs = env.reset()
# for t in range(N):
#     act = policy(obs, sample=False)
#     trs = DotDict(act=act)
#     _, obs = env.step(trs=trs)
# env.env.close()