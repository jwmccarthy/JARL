import numpy as np

from typing import List, Any, Mapping, Generator

from jarl.data.core import MultiList
from jarl.log.progress import Progress


class Logger:

    def __init__(self):
        self.data = MultiList(episode=MultiList())

    def episode(self, t: int, info: Mapping[str, List[Any]]) -> None:
        self.data.episode.extend(info)
        new_info = dict(global_t=t)
        for key, val in self.data.episode.items():
            if info[key]:
                new_info |= {key: np.mean(val[-100:])}
        self.progress_bar.update(episode=new_info)

    def update(self, info: Mapping[str, Any]) -> None:
        self.progress_bar.update(**info)

    def progress(self, steps: int, **kwargs) -> Generator[int, None, None]:
        self.progress_bar = Progress(steps, **kwargs)
        for t in self.progress_bar:
            yield t
        # save logged stats