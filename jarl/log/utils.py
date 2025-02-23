import torch as th

from typing import List

from jarl.data.core import MultiTensor


def get_episodes(data: MultiTensor) -> List[MultiTensor]:
    episodes = []

    for i in range(data.shape[1]):
        d = data[:, i]
        
        # get indices for episode boundaries
        end_idx = th.nonzero(d.don).squeeze().tolist()
        if isinstance(end_idx, int):
            end_idx = [end_idx]
        beg_idx = [0] + [i + 1 for i in end_idx]

        for i, j in zip(beg_idx, end_idx):
            episodes.append(d[i:j+1])

    return episodes


def episodic_return(data: MultiTensor) -> List[float]:
    return [e.rew.sum().item() for e in get_episodes(data)]


def episodic_length(data: MultiTensor) -> List[int]:
    return [len(e) for e in get_episodes(data)]