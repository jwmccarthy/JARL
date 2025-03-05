from typing import Any, Mapping, Generator

from jarl.data.core import MultiList
from jarl.log.progress import Progress


class Logger:

    def __init__(self):
        self.data = MultiList(
            episodic=MultiList()
        )

    def log_transition(self, t: int, trs: Mapping[str, Any]) -> None:
        ...

    def log_update(self, t: int, info: Mapping[str, Any]) -> None:
        ...

    def progress(self, steps: int, **kwargs) -> Generator[int, None, None]:
        self.progress_bar = Progress(steps, **kwargs)
        for t in self.progress_bar:
            yield t
        # save logged stats