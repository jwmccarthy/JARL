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
        self._progress_metric_specs = {}
        self._progress_metric_tasks = {}
        self._global_t_task = None

        self.register_progress_metric("Episode", "reward", format_spec=",.2f")
        self.register_progress_metric(
            "Episode", "historical_reward", format_spec=",.2f"
        )
        self.register_progress_metric(
            "Episode", "length", label="episode_length", format_spec=",.1f"
        )

        if log_dir:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)

    def _write(self, info: Mapping[str, Any], step: int) -> None:
        if not self.writer:
            return

        for section, values in info.items():
            for key, value in values.items():
                self.writer.add_scalar(f"{section}/{key}", float(value), step)

    def register_progress_metric(
        self,
        section: str,
        key: str,
        label: str | None = None,
        format_spec: str = ",.2f",
    ) -> None:
        if not section or not key:
            raise ValueError("progress metric section and key cannot be empty")
        identifier = (section, key)
        if identifier in self._progress_metric_specs:
            raise ValueError(f"progress metric already registered: {section}/{key}")

        self._progress_metric_specs[identifier] = (label or key, format_spec)
        if self._metrics is not None:
            self._progress_metric_tasks[identifier] = self._metrics.add_task(
                label or key,
                total=None,
                value="-",
            )

    def _update_progress_metrics(self, info: Mapping[str, Any]) -> None:
        if self._metrics is None:
            return

        for identifier, task in self._progress_metric_tasks.items():
            section, key = identifier
            values = info.get(section)
            if not isinstance(values, Mapping) or key not in values:
                continue
            _, format_spec = self._progress_metric_specs[identifier]
            self._metrics.update(
                task,
                value=format(float(values[key]), format_spec),
            )

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
        self._update_progress_metrics(update)

    def update(self, info: Mapping[str, Any], step: int = None) -> None:
        if step is not None:
            self.step = step

        self._write(info, self.step)
        self._update_progress_metrics(info)

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
        self._progress = progress
        self._metrics = metrics
        self._global_t_task = global_t_task
        self._progress_metric_tasks = {
            identifier: metrics.add_task(
                label,
                total=None,
                value="-",
            )
            for identifier, (label, _) in self._progress_metric_specs.items()
        }

        try:
            with Live(Group(progress, metrics), refresh_per_second=10):
                yield
        finally:
            self._progress = None
            self._metrics = None
            self._global_t_task = None
            self._progress_metric_tasks = {}

            if self.writer:
                self.writer.close()

    def advance(self, timesteps: int) -> None:
        if self._progress is not None and self._global_t_task is not None:
            self._progress.advance(self._global_t_task, timesteps)
