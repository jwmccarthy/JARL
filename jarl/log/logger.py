import numpy as np

from collections import defaultdict
from typing import Any, Generator, List, Mapping


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
        self._write(update, t)

    def update(self, info: Mapping[str, Any], step: int = None) -> None:
        if step is not None:
            self.step = step

        self._write(info, self.step)

    def progress(self, updates: int) -> Generator[int, None, None]:
        try:
            for update in range(updates):
                yield update
                print(
                    f"\rUpdate {update + 1:,}/{updates:,} | "
                    f"Global ticks {self.step:,}",
                    end="",
                    flush=True,
                )
        finally:
            print()

            if self.writer:
                self.writer.close()
