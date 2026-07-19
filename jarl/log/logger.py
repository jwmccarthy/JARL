import numpy as np

from collections import defaultdict
from typing import List, Any, Mapping, Generator

from jarl.log.progress import Progress


class Logger:
    def __init__(self, log_dir: str = None):
        self.episode_data = defaultdict(list)
        self.step = 0
        self.writer = None
        if log_dir:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)

    def _write(self, info: Mapping[str, Any], step: int) -> None:
        if not self.writer:
            return
        for section, values in info.items():
            for key, value in values.items():
                self.writer.add_scalar(f"{section}/{key}", float(value), step)

    def episode(self, t: int, info: Mapping[str, List[Any]]) -> None:
        self.step = t
        for key, values in info.items():
            self.episode_data[key].extend(values)
        new_info = dict(global_t=t)
        for key, val in self.episode_data.items():
            if info[key]:
                new_info |= {key: np.mean(val[-50:])}
        update = dict(Episode=new_info)
        self.progress_bar.update(**update)
        self._write(update, t)

    def update(self, info: Mapping[str, Any], step: int = None) -> None:
        if step is not None:
            self.step = step
        self.progress_bar.update(**info)
        self._write(info, self.step)

    def progress(self, steps: int, **kwargs) -> Generator[int, None, None]:
        self.progress_bar = Progress(steps, **kwargs)
        try:
            for t in self.progress_bar:
                yield t
        finally:
            if self.writer:
                self.writer.close()
