from __future__ import annotations

from typing import Protocol


class InputType(Protocol):
    @property
    def head(self) -> str: ...
