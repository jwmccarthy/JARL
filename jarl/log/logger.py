import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from collections import defaultdict
from typing import Any, Generator, List, Mapping


class Logger:
    def __init__(self, log_dir: str = None):
        self.episode_data = defaultdict(list)
        self.step = 0
        self.writer = None
        self._progress = None
        self._updates_task = None

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

        if self._progress is not None and self._updates_task is not None:
            self._progress.advance(self._updates_task)

    def progress(
        self,
        vector_steps: int,
        environments_per_step: int,
        learner_updates: int,
    ) -> Generator[int, None, None]:
        progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed:,.0f}/{task.total:,.0f}"),
        )
        global_t_task = progress.add_task(
            "global_t",
            total=vector_steps * environments_per_step,
        )
        updates_task = progress.add_task("updates", total=learner_updates)
        self._progress = progress
        self._updates_task = updates_task

        try:
            with progress:
                for vector_step in range(vector_steps):
                    yield vector_step
                    progress.advance(global_t_task, environments_per_step)
        finally:
            self._progress = None
            self._updates_task = None

            if self.writer:
                self.writer.close()
