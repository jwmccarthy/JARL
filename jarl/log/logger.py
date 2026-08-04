from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text


MetricIdentifier = tuple[str, str]
Metrics = Mapping[str, Mapping[str, Any]]


class ScalarWriter(Protocol):

    def add_scalar(
        self,
        tag: str,
        scalar_value: Any,
        global_step: int,
    ) -> None: ...

    def flush(self) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True, slots=True)
class ProgressMetricSpec:
    label:       str
    format_spec: str


class AverageTimeRemainingColumn(ProgressColumn):

    def render(self, task: Task) -> Text:
        if task.total is None or task.completed <= 0 or task.elapsed is None:
            return Text("--:--:--", style="progress.remaining")

        remaining = max(task.total - task.completed, 0.0)
        seconds = round(task.elapsed * remaining / task.completed)

        return Text(
            str(timedelta(seconds=seconds)),
            style="progress.remaining",
        )


class Logger:
    def __init__(
        self,
        log_dir: str | Path | None = None,
    ) -> None:
        self.step = 0
        self._writer: ScalarWriter | None = self._create_writer(log_dir)

        self._progress: Progress | None = None
        self._metrics: Progress | None = None
        self._main_task: TaskID | None = None
        self._activity_task: TaskID | None = None

        self._progress_metric_specs: dict[
            MetricIdentifier,
            ProgressMetricSpec,
        ] = {}
        self._progress_metric_tasks: dict[
            MetricIdentifier,
            TaskID,
        ] = {}

    @staticmethod
    def _create_writer(
        log_dir: str | Path | None,
    ) -> ScalarWriter | None:
        if log_dir is None:
            return None

        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(str(log_dir))

    def register_progress_metric(
        self,
        section:     str,
        key:         str,
        label:       str | None = None,
        format_spec: str = ",.2f",
    ) -> None:
        if not section or not key:
            raise ValueError(
                "progress metric section and key cannot be empty"
            )

        identifier = (section, key)

        if identifier in self._progress_metric_specs:
            raise ValueError(
                f"progress metric already registered: {section}/{key}"
            )

        spec = ProgressMetricSpec(
            label=label or key,
            format_spec=format_spec,
        )
        self._progress_metric_specs[identifier] = spec

        if self._metrics is not None:
            self._progress_metric_tasks[identifier] = (
                self._create_metric_task(spec)
            )

    def update(
        self,
        metrics: Metrics,
        step:    int | None = None,
    ) -> None:
        if step is not None:
            self.step = step

        self._write(metrics)
        self._update_progress_metrics(metrics)

    def _write(self, metrics: Metrics) -> None:
        if self._writer is None:
            return

        for section, values in metrics.items():
            for key, value in values.items():
                self._writer.add_scalar(
                    f"{section}/{key}",
                    float(value),
                    self.step,
                )

    def _update_progress_metrics(self, metrics: Metrics) -> None:
        if self._metrics is None:
            return

        for identifier, task_id in self._progress_metric_tasks.items():
            section, key = identifier
            section_metrics = metrics.get(section)

            if section_metrics is None or key not in section_metrics:
                continue

            spec = self._progress_metric_specs[identifier]
            value = format(
                float(section_metrics[key]),
                spec.format_spec,
            )

            self._metrics.update(task_id, value=value)

    @contextmanager
    def progress(
        self,
        total_steps:   int,
        initial_steps: int = 0,
        description:   str = "progress",
    ) -> Generator[None, None, None]:
        self._validate_progress(total_steps, initial_steps)

        if self._progress is not None:
            raise RuntimeError("a progress display is already active")

        self._initialize_progress(
            total_steps=total_steps,
            initial_steps=initial_steps,
            description=description,
        )

        assert self._progress is not None
        assert self._metrics is not None

        try:
            with Live(
                Group(self._progress, self._metrics),
                refresh_per_second=10,
            ):
                yield
        finally:
            self._clear_progress()

    @staticmethod
    def _validate_progress(
        total_steps:   int,
        initial_steps: int,
    ) -> None:
        if total_steps < 0:
            raise ValueError("total_steps cannot be negative")

        if not 0 <= initial_steps <= total_steps:
            raise ValueError(
                "initial_steps must be between zero and total_steps"
            )

    def _initialize_progress(
        self,
        total_steps:   int,
        initial_steps: int,
        description:   str,
    ) -> None:
        self._progress = self._create_progress_display()
        self._metrics = self._create_metrics_display()

        self._main_task = self._progress.add_task(
            description,
            total=total_steps,
            completed=initial_steps,
        )
        self._activity_task = self._progress.add_task(
            "idle",
            total=1,
            completed=0,
            start=False,
        )

        self._progress_metric_tasks = {
            identifier: self._create_metric_task(spec)
            for identifier, spec in self._progress_metric_specs.items()
        }

    @staticmethod
    def _create_progress_display() -> Progress:
        return Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed:,.0f}/{task.total:,.0f}"),
            TimeElapsedColumn(),
            TextColumn("<"),
            AverageTimeRemainingColumn(),
            auto_refresh=False,
        )

    @staticmethod
    def _create_metrics_display() -> Progress:
        return Progress(
            TextColumn("[bold]{task.description}"),
            TextColumn("{task.fields[value]}"),
            auto_refresh=False,
        )

    def _create_metric_task(
        self,
        spec: ProgressMetricSpec,
    ) -> TaskID:
        if self._metrics is None:
            raise RuntimeError("metrics display is not active")

        return self._metrics.add_task(
            spec.label,
            total=None,
            value="-",
        )

    def _clear_progress(self) -> None:
        self._progress = None
        self._metrics = None
        self._main_task = None
        self._activity_task = None
        self._progress_metric_tasks.clear()

    def advance(self, amount: int = 1) -> None:
        if self._progress is None or self._main_task is None:
            return

        self._progress.advance(self._main_task, amount)

    def start_activity(
        self,
        total:       int,
        description: str,
    ) -> None:
        if total < 0:
            raise ValueError("activity total cannot be negative")

        if self._progress is None or self._activity_task is None:
            return

        self._progress.reset(
            self._activity_task,
            description=description,
            total=total,
            completed=0,
            start=True,
        )

    def advance_activity(self, amount: int = 1) -> None:
        if self._progress is None or self._activity_task is None:
            return

        self._progress.advance(self._activity_task, amount)

    def finish_activity(self) -> None:
        if self._progress is None or self._activity_task is None:
            return

        self._progress.reset(
            self._activity_task,
            description="idle",
            total=1,
            completed=0,
            start=False,
        )

    def start(
        self,
        total:   int,
        section: str = "Update",
    ) -> None:
        self.start_activity(total, section)

    def epoch_finished(self) -> None:
        self.advance_activity()

    def finish(self) -> None:
        self.finish_activity()

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is None:
            return

        self._writer.close()
        self._writer = None

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()