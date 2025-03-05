import torch as th

import time
from typing import List

from jarl.data.dict import DotDict
from jarl.data.buffer import Buffer

from jarl.envs.env import SyncEnv
from jarl.modules.policy import Policy
from jarl.train.graph import TrainGraph
from jarl.log.progress import Progress


class TrainLoop:

    def __init__(
        self, 
        env: SyncEnv, 
        buffer: Buffer,
        policy: Policy,
        graphs: List[TrainGraph],
        logger: Progress = Progress
    ) -> None:
        self.env = env
        self.buffer = buffer
        self.policy = policy
        self.logger = logger
        self.graphs = graphs

    def ready(self, t: int) -> List[TrainGraph]:
        return [g for g in self.graphs if g.ready(t)]

    def run(self, steps: int) -> None:
        obs = self.env.reset()
        log = self.logger(steps)
        global_t = 0

        for g in self.graphs:
            g.init_schedulers(steps)

        start_time = time.time()

        for t in log:
            # step environment
            with th.no_grad():
                act = self.policy(obs)
            trs = DotDict(obs=obs, act=act)
            trs, obs = self.env.step(trs=trs)

            global_t += self.env.n_envs

            # store data
            self.buffer.store(trs)

            # log transition
            # log.log_transition(trs)
            log.update(time=dict(
                steps=global_t,
                elapsed=(time.time() - start_time) / 60
            ))

            # run blocks
            for graph in self.ready(t):
                data = self.buffer.serve()
                info = graph.update(data)
                log.update(**info)

        return log