import numpy as np
import torch as th

import gymnasium as gym
from typing import Tuple, Callable, Any, Self

from jarl.data.dict import DotDict
from jarl.data.types import Device, EnvOutput
from jarl.envs.space import torch_space


class TorchEnv:

    def __init__(
        self, 
        env_id: str, 
        device: Device = "cpu",
        **kwargs: dict
    ) -> None:
        self.env = gym.make(env_id, **kwargs)
        self.device = device

        # wrap spaces for tensor spec
        self.obs_space = torch_space(
            self.env.observation_space, device)
        self.act_space = torch_space(
            self.env.action_space, device)
        
        # track episode stats
        self.reward = self.length = 0

    def seed(self, seed: int) -> None:
        self.env.seed(seed)

    def reset(self) -> Tuple[th.Tensor, dict]:
        obs, _ = self.env.reset()
        return th.as_tensor(obs).to(self.device)

    def step(self, act: np.ndarray) -> Tuple[th.Tensor, ...]:
        *exp, _ = self.env.step(act)

        # conver to tensors
        nxt, rew, trm, trc = [
            th.tensor(x, device=self.device) for x in exp]
        don = th.logical_or(trm, trc)

        self.reward += rew.item()
        self.length += 1

        # auto-reset
        if don:
            obs = self.reset()
            info = DotDict(reward=self.reward, length=self.length)
            self.reward = self.length = 0
        else:
            obs, info = nxt, {}

        return nxt, rew, trc, don, obs, info


class SyncEnv:

    def __init__(
        self, 
        env_id: str, 
        n_envs: int = 1, 
        device: Device = "cpu",
        **kwargs: dict
    ) -> None:
        self.n_envs = n_envs
        self.device = device
        self.rews, self.lens = [], []
        self.envs = [TorchEnv(env_id, **kwargs) for _ in range(n_envs)]
        
        # wrap obs/act space
        self.obs_space = self.envs[0].obs_space
        self.act_space = self.envs[0].act_space

        # storage for transition values
        self.exp = DotDict(
            nxt=th.empty((self.n_envs, *self.obs_space.shape)),
            rew=th.empty((self.n_envs,)),
            trc=th.empty((self.n_envs,), dtype=bool),
            don=th.empty((self.n_envs,), dtype=bool)
        )
        self.obs = th.empty_like(self.exp.nxt)

    def reset(self) -> th.Tensor:
        obs = np.stack([env.reset()[0] for env in self.envs])
        return th.tensor(obs, device=self.device)
    
    def step(self, trs: DotDict[str, th.Tensor]) -> EnvOutput:
        lens, rews = [], []

        # back to numpy to step env
        actions = trs.act.detach().cpu().numpy()

        # step environments
        for i, (env, act) in enumerate(zip(self.envs, actions)):
            *exp, obs, info = env.step(act)
            for key, val in zip(self.exp.keys(), exp):
                self.exp[key][i] = val
            self.obs[i] = obs

            # obtain episode stats
            if info:
                lens.append(info.length)
                rews.append(info.reward)

        trs.update(self.exp)

        # get ep stats
        trs.info = dict(length=lens, reward=rews)

        return trs, self.obs