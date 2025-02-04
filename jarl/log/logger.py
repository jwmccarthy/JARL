import numpy as np
import torch as th

from typing import Dict, Any
from collections import deque

from jarl.data.multi import MultiTensor
from jarl.log.utils import get_episodes
from jarl.log.progress import ProgressBar


class Logger:

    def __init__(
        self,
        window_size: int = 100
    ) -> None:
        self.window_size = window_size
        self.ep_rets = deque(maxlen=self.window_size)
        self.ep_lens = deque(maxlen=self.window_size)

    def log_data(self, data: MultiTensor) -> None:
        episodes = get_episodes(data)

        # log episodic metrics
        if "raw_rew" in data:
            self.ep_rets.extend([e.raw_rew.sum().item() for e in episodes])
        else:
            self.ep_rets.extend([e.rew.sum().item() for e in episodes])
        self.ep_lens.extend([len(e) for e in episodes])

        # update progress bar
        if self.bar is not None:
            self.bar.update("Episode", {
                "mean return": np.mean(self.ep_rets),
                "mean length": np.mean(self.ep_lens)
            })

    def log_info(self, info: Dict[str, th.Tensor]) -> None:
        self.bar.update("Update", info)

    def progress(
        self, 
        steps: int,
        **kwargs: Dict[str, Any]
    ) -> ProgressBar:
        self.bar = ProgressBar(steps, **kwargs)
        return self.bar
    
    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
        self.ep_rets.clear()
        self.ep_lens.clear()