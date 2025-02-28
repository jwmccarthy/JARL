import torch as th

from collections import defaultdict
from typing import Any, Self, Dict, List

from jarl.data.core import MultiTensor
from jarl.train.update.base import ModuleUpdate
from jarl.train.modify.base import DataModifier
from jarl.train.sample.base import Sampler, IdentitySampler

from jarl.train.utils import (
    all_combinations, 
    compose_funcs,
    topological_sort
)


# we can assume these keys will come from the buffer
ROOT = {"obs", "act", "rew", "don", "nxt", "trc"}


class TrainGraph:

    def __init__(
        self, 
        updates: ModuleUpdate | List[ModuleUpdate],
        sampler: Sampler = IdentitySampler()
    ) -> None:
        self.sampler = sampler

        if not isinstance(updates, list):
            updates = [updates]
        self.updates = updates

        # dependency graph construction
        self.mod_graph = defaultdict(set)
        self.update_dep = []    # functionals for each update combo
        self.active_dep = None  # active dependencies for ready updates
        self.update_queue = []  # queue for ready updates

    def add_modifier(self, new_mod: DataModifier) -> Self:
        self.mod_graph[new_mod]  # add if not present

        # build dependency graph
        for mod in self.mod_graph.keys():

            # incoming mod -> stored mod
            if new_mod.requires_keys & mod.produces_keys:
                self.mod_graph[mod].add(new_mod)

            # stored mod -> incoming mod
            if new_mod.produces_keys & mod.requires_keys:
                self.mod_graph[new_mod].add(mod)

        return self

    def compile(self) -> Self:
        # topological sort for all dependencies
        global_top = topological_sort(self.mod_graph)

        # isolate sub-graphs for each update
        for updates in all_combinations(self.updates):
            current_deps = []
            
            # init required keys for update combo
            prd_keys, req_keys = set(), set()
            for update in updates:
                req_keys |= update.requires_keys

            # obtain deps for current update
            for m in global_top:
                if m.produces_keys & req_keys:
                    current_deps.append(m)
                    req_keys |= m.requires_keys
                    prd_keys |= m.produces_keys

            # check all dependencies met
            missing = req_keys - prd_keys - ROOT
            assert not missing, f"Missing keys {missing}"

            # nested functions for each update
            func = compose_funcs(current_deps)
            self.update_dep.append(func)

        del self.mod_graph  # graph not needed after compilation

        return self
    
    def init_schedulers(self, steps: int) -> None:
        for update in self.updates:
            up_step = steps // update.freq
            update.scheduler.start(up_step)
    
    def ready(self, t: int) -> bool:
        mod_idx = 0

        # get ready updates & corresponding deps 
        for i, update in enumerate(self.updates):
            if update.ready(t):
                mod_idx += 1 << i
                self.update_queue.append(update)

        self.active_dep = self.update_dep[mod_idx]

        return bool(self.update_queue)
    
    def update(self, data: MultiTensor) -> Dict[str, Any]:
        data = self.active_dep(data)

        for batch in self.sampler.sample(data):
            batch_info = {}
            for update in self.update_queue:
                batch_info |= update(batch)

        for update in self.update_queue:
            if update.scheduler:
                update.scheduler.step()

        self.update_queue = []  # reset update queue
        
        return batch_info
