import torch.nn as nn
from torch.optim import Adam

import gymnasium as gym

from jarl.envs.vec import TorchGymEnv
from jarl.envs.wrap import FormatImageWrapper, ObsStackWrapper

from jarl.data.dict import DotDict
from jarl.data.buffer import LazyBuffer

from jarl.modules.core import MLP
from jarl.modules.operator import Critic
from jarl.modules.policy import CategoricalPolicy
from jarl.modules.encoder.core import FlattenEncoder

from jarl.train.optim import Optimizer
from jarl.train.update.ppo import PPOUpdate
from jarl.train.sample.batch import BatchSampler

from jarl.train.loop import TrainLoop
from jarl.train.graph import TrainGraph

from jarl.train.modify.compute import (
    ComputeValues,
    ComputeLogProbs,
    ComputeAdvantages,
    ComputeReturns,
)


env = TorchGymEnv("LunarLander-v2", 4)
# env = FormatImageWrapper(env)
# env = ObsStackWrapper(env, 4)

policy = CategoricalPolicy(
    head=FlattenEncoder(),
    body=MLP(func=nn.Tanh, dims=[64, 64])
).build(env)

critic = Critic(
    head=FlattenEncoder(), 
    body=MLP(func=nn.Tanh, dims=[64, 64]),
).build(env)

ppo = (
    TrainGraph(
        PPOUpdate(2048, policy, critic, optimizer=Optimizer(Adam, lr=3e-4)),
        BatchSampler(64, num_epoch=10)
    )
    .add_modifier(ComputeAdvantages())
    .add_modifier(ComputeLogProbs(policy))
    .add_modifier(ComputeReturns())
    .add_modifier(ComputeValues(critic))
    .compile()
)

buffer = LazyBuffer(2048)

loop = TrainLoop(env, buffer, policy, graphs=[ppo])
loop.run(int(1e6))


env = TorchGymEnv("LunarLander-v2")

N = 16384

obs = env.reset()
for t in range(N):
    act = policy(obs, sample=False)
    trs = DotDict(obs=obs, act=act)
    trs, obs = env.step(trs=trs)
env.env.close()