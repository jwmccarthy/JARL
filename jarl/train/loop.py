import torch as th

import time
from typing import List

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
        logger: Logger = Logger()
    ) -> None:
        self.env = env
        self.buffer = buffer
        self.policy = policy
        self.logger = logger
        self.graphs = graphs

    def ready(self, t: int) -> List[TrainGraph]:
        return [g for g in self.graphs if g.ready(t)]

    def run(self, steps: int) -> None:
        global_t = 0
        obs = self.env.reset()

        for g in self.graphs:
            g.init_schedulers(steps)

        for t in self.logger.progress(steps):
            # step environment
            with th.no_grad():
                act = self.policy(obs)
            trs = DotDict(obs=obs, act=act)
            trs, obs = self.env.step(trs=trs)

            global_t += self.env.n_envs

            # track episode info
            self.logger.episode(global_t, trs.pop("info"))

            # store data
            self.buffer.store(trs)

            # run blocks
            for graph in self.ready(t):
                data = self.buffer.serve()
                info = graph.update(data)
                self.logger.update(info)

        return self.policy