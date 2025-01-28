from functools import reduce
from itertools import combinations
from typing import Any, Dict, Set, List


def all_combinations(iterable):
    for i in range(1, len(iterable) + 1):
        yield from combinations(iterable, i)


def compose_funcs(functions):
    return reduce(
        lambda agg, f: lambda data: agg(f(data)),
        functions,
        lambda x: x
    )


def topological_sort(graph: Dict[Any, Set[Any]]) -> List[Any]:

    def _sort_util(
        node: Any, 
        visited: Dict[Any, bool], 
        stack: List[Any]
    ) -> None:
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                _sort_util(neighbor, visited, stack)
        stack.append(node)

    out = []
    vis = {k: False for k in graph.keys()}

    for node in graph.keys():
        if not vis[node]:
            _sort_util(node, vis, out)

    return out