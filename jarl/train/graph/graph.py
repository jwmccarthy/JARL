from typing import Any, Self, Dict
from collections import defaultdict

from jarl.data.multi import MultiTensor
from jarl.train.sample.base import BatchSampler
from jarl.train.update.base import ModuleUpdate
from jarl.train.modify.base import DataModifier

from jarl.train.graph.utils import (
    all_combinations, 
    compose_funcs,
    topological_sort
)


class TrainGraph:

    def __init__(
        self, 
        sampler: BatchSampler,
        *updates: ModuleUpdate
    ) -> None:
        self.sampler = sampler
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
            requires_keys = set()
            for update in updates:
                requires_keys |= update.requires_keys

            # obtain deps for current update
            for m in global_top:
                if m.produces_keys & requires_keys:
                    current_deps.append(m)
                    requires_keys |= m.requires_keys

            # nested functions for each update
            # func = compose_funcs(current_deps[::-1])
            # self.update_dep.append(func)
            self.update_dep.append(current_deps[::-1])

        del self.mod_graph  # graph not needed after compilation

        return self
    
    def ready(self, t: int) -> bool:
        mod_idx = -1

        # get ready updates & corresponding deps 
        for i, update in enumerate(self.updates):
            if update.ready(t):
                mod_idx += 1 << i
                self.update_queue.append(update)

        # set active dependency function if ready
        if (is_ready := bool(self.update_queue)):
            self.active_dep = self.update_dep[mod_idx]

        return is_ready
    
    def __call__(self, data: MultiTensor) -> Dict[str, Any]:
        data = self.active_dep(data)

        for batch in self.sampler(data):
            batch_info = {}
            for update in self.update_queue:
                batch_info |= update(batch)

        self.update_queue = []  # reset update queue
        
        return batch_info