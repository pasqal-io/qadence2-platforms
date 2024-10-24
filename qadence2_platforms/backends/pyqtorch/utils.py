from __future__ import annotations

from typing import Any, Protocol


class InputType(Protocol):
    @property
    def head(self) -> InputType: ...

    @property
    def args(self) -> InputType: ...

    @property
    def value(self) -> Any: ...

    def is_symbol(self) -> bool: ...

    def is_quantum_operator(self) -> bool: ...

    def __getitem__(self, item: slice | int) -> Any: ...
