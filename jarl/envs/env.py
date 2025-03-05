import numpy as np
import torch as th

import gymnasium as gym
from typing import Tuple, Callable, Any

from jarl.data.dict import DotDict
from jarl.data.types import Device, EnvOutput
from jarl.envs.space import torch_space


class TorchEnv:

    def __init__(self, env: gym.Env, device: Device = "cpu") -> None:
        self.env = env
        self.device = device

        # wrap spaces for tensor spec
        self.obs_space = torch_space(
            self.env.observation_space, device)
        self.act_space = torch_space(
            self.env.action_space, device)

    def seed(self, seed: int) -> None:
        self.env.seed(seed)

    def reset(self) -> Tuple[th.Tensor, dict]:
        obs, _ = self.env.reset()
        return th.as_tensor(obs).to(self.device)
    
    def step(self, act: th.Tensor) -> Tuple[th.Tensor, ...]:
        act = act.detach().cpu().numpy()
        *exp, _ = self.env.step(act)
        obs, rew, trm, trc = [
            th.as_tensor(x).to(self.device) for x in exp]
        return obs, rew, trm, trc


class SyncEnv:

    def __init__(
        self, 
        env_func: Callable[[Any], gym.Env], 
        n_envs: int = 1, 
        device: Device = "cpu"
    ) -> None:
        self.n_envs = n_envs
        self.device = device
        self.envs = [TorchEnv(env_func(), device) for _ in range(n_envs)]
        self.obs_space = self.envs[0].obs_space
        self.act_space = self.envs[0].act_space

    def reset(self) -> th.Tensor:
        obs = [env.reset() for env in self.envs]
        return th.stack(obs)
    
    def step(self, trs: DotDict[str, th.Tensor]) -> EnvOutput:
        act = trs.act  # get action from active transition

        # stack transition from each env
        exp = zip(*[env.step(a) for a, env in zip(act, self.envs)])
        obs, rew, trm, trc = [th.stack(x) for x in exp]
        don = th.logical_or(trm, trc)

        # autoreset
        nxt = obs.clone()
        idx = don.nonzero(as_tuple=True)[0]
        if idx.numel():
            res = [self.envs[i].reset() for i in idx]
            nxt[idx] = th.stack(res)

        trs.update(nxt=obs, rew=rew, trc=trc, don=don)

        return trs, nxt
