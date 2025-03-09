import numpy as np

from typing import Any, Mapping, Generator

from jarl.data.core import MultiList
from jarl.log.progress import Progress


class Logger:

    def __init__(self):
        self.data = MultiList(episode=MultiList())

    def episode(self, t: int, info: Mapping[str, Any]) -> None:
        self.data.episode.extend(info)
        for key, val in self.data.episode.items():
            stat_mean = np.mean(val[-100:])
            self.progress_bar.update(episode={key: stat_mean})
        self.progress_bar.update(episode=dict(global_t=t))

    def update(self, info: Mapping[str, Any]) -> None:
        self.progress_bar.update(**info)

    def progress(self, steps: int, **kwargs) -> Generator[int, None, None]:
        self.progress_bar = Progress(steps, **kwargs)
        for t in self.progress_bar:
            yield t
        # save logged stats