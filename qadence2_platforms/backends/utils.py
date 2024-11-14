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

    @property
    def is_symbol(self) -> bool: ...

    @property
    def is_quantum_operator(self) -> bool: ...

    @property
    def is_addition(self) -> bool: ...

    @property
    def is_multiplication(self) -> bool: ...

    @property
    def is_kronecker_product(self) -> bool: ...

    @property
    def is_power(self) -> bool: ...

    @property
    def subspace(self) -> Support | None: ...

    def add(self, *args: InputType) -> InputType: ...

    def mul(self, *args: InputType) -> InputType: ...

    def kron(self, *args: InputType) -> InputType: ...

    def pow(self, *args: InputType) -> InputType: ...

    def __getitem__(self, item: slice | int) -> Any: ...
