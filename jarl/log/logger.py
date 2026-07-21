import numpy as np
from contextlib import contextmanager
from datetime import timedelta
from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

from collections import defaultdict, deque
from typing import Any, Generator, List, Mapping


class AverageTimeRemainingColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        if task.total is None or task.completed <= 0 or task.elapsed is None:
            return Text("--:--:--", style="progress.remaining")
        remaining = max(task.total - task.completed, 0.0)
        seconds = round(task.elapsed * remaining / task.completed)
        return Text(str(timedelta(seconds=seconds)), style="progress.remaining")


class Logger:
    def __init__(self, log_dir: str = None):
        self.episode_data = defaultdict(lambda: deque(maxlen=50))
        self.step = 0
        self.writer = None
        self._progress = None
        self._metrics = None
        self._reward_task = None
        self._historical_reward_task = None
        self._imitation_reward_task = None
        self._episode_length_task = None
        self._global_t_task = None

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

        for key, values in self.episode_data.items():
            if info.get(key):
                new_info |= {key: np.mean(values)}

        update = dict(Episode=new_info)
        self._write(update, t)

        if self._metrics is not None:
            if "reward" in new_info:
                self._metrics.update(
                    self._reward_task,
                    value=f"{new_info['reward']:,.2f}",
                )

            if "length" in new_info:
                self._metrics.update(
                    self._episode_length_task,
                    value=f"{new_info['length']:,.1f}",
                )
            if "historical_reward" in new_info:
                self._metrics.update(
                    self._historical_reward_task,
                    value=f"{new_info['historical_reward']:,.2f}",
                )

    def update(self, info: Mapping[str, Any], step: int = None) -> None:
        if step is not None:
            self.step = step

        self._write(info, self.step)
        imitation_reward = info.get("GAIfO", {}).get("imitation_reward")
        if imitation_reward is not None and self._metrics is not None:
            self._metrics.update(
                self._imitation_reward_task,
                value=f"{imitation_reward:,.4f}",
            )

    @contextmanager
    def progress(self, total_timesteps: int) -> Generator[None, None, None]:
        progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed:,.0f}/{task.total:,.0f}"),
            TimeElapsedColumn(),
            TextColumn("<"),
            AverageTimeRemainingColumn(),
            auto_refresh=False,
        )
        metrics = Progress(
            TextColumn("[bold]{task.description}"),
            TextColumn("{task.fields[value]}"),
            auto_refresh=False,
        )
        global_t_task = progress.add_task(
            "global_t",
            total=total_timesteps,
        )
        reward_task = metrics.add_task(
            "reward",
            total=None,
            value="-",
        )
        historical_reward_task = metrics.add_task(
            "historical_reward",
            total=None,
            value="-",
        )
        imitation_reward_task = metrics.add_task(
            "imitation_reward",
            total=None,
            value="-",
        )
        episode_length_task = metrics.add_task(
            "episode_length",
            total=None,
            value="-",
        )
        self._progress = progress
        self._metrics = metrics
        self._global_t_task = global_t_task
        self._reward_task = reward_task
        self._historical_reward_task = historical_reward_task
        self._imitation_reward_task = imitation_reward_task
        self._episode_length_task = episode_length_task

        try:
            with Live(Group(progress, metrics), refresh_per_second=10):
                yield
        finally:
            self._progress = None
            self._metrics = None
            self._global_t_task = None
            self._reward_task = None
            self._historical_reward_task = None
            self._imitation_reward_task = None
            self._episode_length_task = None

            if self.writer:
                self.writer.close()

    def advance(self, timesteps: int) -> None:
        if self._progress is not None and self._global_t_task is not None:
            self._progress.advance(self._global_t_task, timesteps)
