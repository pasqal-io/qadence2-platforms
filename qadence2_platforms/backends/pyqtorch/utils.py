from __future__ import annotations

from typing import Any, Protocol


class Support(Protocol):
    def target_all(self) -> Any: ...

    @property
    def target(self) -> list[int]: ...

    @property
    def control(self) -> list[int]: ...


class InputType(Protocol):
    @property
    def head(self) -> InputType | str: ...

    @property
    def args(self) -> InputType | list[InputType | str | Support]: ...

    @property
    def value(self) -> Any: ...

    def is_symbol(self) -> bool: ...

    def is_quantum_operator(self) -> bool: ...

    def __getitem__(self, item: slice | int) -> Any: ...
