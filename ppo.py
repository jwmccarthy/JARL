import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

import gymnasium as gym

from jarl.envs.gym import SyncEnv, TorchEnv

from jarl.data.buffer import LazyBuffer

from jarl.modules.core import MLP, CNN
from jarl.modules.operator import Critic
from jarl.modules.policy import CategoricalPolicy
from jarl.modules.encoder.image import ImageEncoder
from jarl.modules.encoder.core import FlattenEncoder

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

from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from jarl.envs.wrappers import (
    NoopResetWrapper,
    MaxAndSkipWrapper,
    EpisodicLifeWrapper,
    FireResetWrapper,
    ImageTransformWrapper,
    FrameStackWrapper
)


# eid = "ale_py:ALE/Breakout-v5"
eid = "CartPole-v1"
env = gym.make(eid)
env = TorchEnv(env, device="cuda")
# env = NoopResetWrapper(env, noop_max=30)
# env = MaxAndSkipWrapper(env, skip=4)
# env = EpisodicLifeWrapper(env)
# env = FireResetWrapper(env)
# env = ImageTransformWrapper(env)
# env = FrameStackWrapper(env, 4)
env = SyncEnv(env, 4, device="cuda")

policy = CategoricalPolicy(
    head=FlattenEncoder(),
    body=MLP()
).build(env).to("cuda")

critic = Critic(
    head=policy.head,    
    body=MLP(),
).build(env).to("cuda")

ppo = (
    TrainGraph(
        PPOUpdate(128, policy, critic, clip=0.2,
                  optimizer=Optimizer(Adam, lr=2.5e-4)),
                #   scheduler=Scheduler(LinearLR, start_factor=1.0, end_factor=0.0)),
        BatchSampler(128, num_epoch=4)
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