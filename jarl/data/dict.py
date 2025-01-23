from typing import Any


class DotDict(dict):
    """Python dictionary w/ dot access"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __getattr__(self, key: Any) -> Any:
        return self[key]
    
    def __setattr__(self, key: Any, value: Any) -> None:
        self[key] = value