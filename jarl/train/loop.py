import numpy as np
import torch as th

import time
from typing import List
from numpy.typing import NDArray

from jarl.data.dict import DotDict
from jarl.data.buffer import Buffer

from jarl.envs.gym import SyncEnv
from jarl.modules.policy import Policy
from jarl.train.graph import TrainGraph
from jarl.log.logger import Logger


class TrainLoop:

    def __init__(
        self, 
        env: SyncEnv, 
        buffer: Buffer,
        policy: Policy,
        graphs: List[TrainGraph],
        logger: Logger = Logger(),
        warmup: int = 0
    ) -> None:
        self.env = env
        self.buffer = buffer
        self.policy = policy
        self.logger = logger
        self.graphs = graphs
        self.warmup = warmup
    
    def _get_action(self, obs: NDArray | th.Tensor, warmup: bool = False) -> th.Tensor:
        if warmup:
            return self.env.sample()
        with th.no_grad():
            obs = th.tensor(obs, device=self.policy.device)
            act = self.policy(obs)
        return act

    def ready(self, t: int) -> List[TrainGraph]:
        return [g for g in self.graphs if g.ready(t)]

    def run(self, steps: int) -> None:
        global_t = 0
        obs = self.env.reset()

        # prepare lr schedulers
        for g in self.graphs:
            g.init_schedulers(steps)
        
        for t in self.logger.progress(steps):
            trs = dict(obs=obs)

            # step environment
            warmup = global_t < self.warmup
            act = self._get_action(obs, warmup=warmup)
            exp, obs, info = self.env.step(act)

            global_t += self.env.n_envs

            # track episode info
            self.logger.episode(global_t, info)

            # store data
            self.buffer.store(trs | exp)

            if self.warmup:
                continue

            # run blocks
            for graph in self.ready(t):
                data = self.buffer.serve()
                info = graph.update(data)
                self.logger.update(info)

        return self.policy