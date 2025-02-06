import torch as th

from typing import Self, List

from jarl.data.dict import DotDict
from jarl.data.buffer import Buffer
from jarl.envs.gym import TorchGymEnv
from jarl.modules.policy import Policy
from jarl.train.graph import TrainGraph
from jarl.log.logger import Logger


class TrainLoop:

    def __init__(
        self, 
        env: TorchGymEnv, 
        buffer: Buffer,
        policy: Policy,
        logger: Logger = Logger(),
        graphs: List[TrainGraph] = []
    ) -> None:
        self.env = env
        self.buffer = buffer
        self.policy = policy
        self.logger = logger
        self.graphs = graphs

    def add_graph(self, graph: TrainGraph) -> Self:
        self.graphs.append(graph)
        return self

    def run(self, steps: int) -> None:
        obs = self.env.reset()
        bar = self.logger.progress(steps)

        for t in bar:
            # step environment
            with th.no_grad():
                act = self.policy(obs)
            trs = DotDict(obs=obs, act=act)
            trs, obs = self.env.step(trs=trs)

            # store data
            self.buffer.store(trs)

            # run blocks
            for graph in self.graphs:
                if not graph.ready(t):
                    continue
                data = self.buffer.serve()
                self.logger.log_data(data)
                info = graph.update(data)
                self.logger.log_info(info)

        # do some tidying up here
        self.logger.close()