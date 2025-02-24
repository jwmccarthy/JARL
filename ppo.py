import torch.nn as nn
from torch.optim import Adam

import gymnasium as gym

from jarl.envs.vec import TorchGymEnv

from jarl.data.dict import DotDict
from jarl.data.buffer import LazyBuffer

from jarl.modules.core import MLP, CNN
from jarl.modules.operator import Critic
from jarl.modules.policy import CategoricalPolicy
from jarl.modules.encoder.image import ImageEncoder

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


def make_env(id):
    def _make_env():
        env = gym.make(id, frameskip=1)
        env = gym.wrappers.AtariPreprocessing(env, frame_skip=4)
        env = gym.wrappers.FrameStack(env, 4)
        return env
    return _make_env

env = TorchGymEnv(make_env("ALE/Breakout-v5"), 8, device="cuda")

policy = CategoricalPolicy(
    head=ImageEncoder(CNN(
        dims=[32, 64, 64],
        kernel=[8, 4, 3],
        stride=[4, 2, 1]
    )),
    body=MLP(func=nn.ReLU, dims=[512])
).build(env).to("cuda")

critic = Critic(
    head=ImageEncoder(CNN(
        dims=[32, 64, 64],
        kernel=[8, 4, 3],
        stride=[4, 2, 1]
    )),    
    body=MLP(func=nn.ReLU, dims=[512]),
).build(env).to("cuda")

ppo = (
    TrainGraph(
        PPOUpdate(128, policy, critic, clip=0.1,
                  optimizer=Optimizer(Adam, lr=2.5e-4)),
        BatchSampler(256, num_epoch=4)
    )
    .add_modifier(ComputeAdvantages())
    .add_modifier(ComputeLogProbs(policy))
    .add_modifier(ComputeReturns())
    .add_modifier(ComputeValues(critic))
    .compile()
)

buffer = LazyBuffer(128).to("cuda")

loop = TrainLoop(env, buffer, policy, graphs=[ppo])
loop.run(int(1e6))


env = TorchGymEnv("ALE/Breakout-v5", render_mode="human")

policy = policy.to("cpu")

N = 16384

obs = env.reset()
for t in range(N):
    act = policy(obs, sample=False)
    trs = DotDict(obs=obs, act=act)
    trs, obs = env.step(trs=trs)
env.env.close()