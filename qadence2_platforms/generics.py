from __future__ import annotations

from typing import Any, Generic, Iterable, Protocol, TypeVar

ExprType = TypeVar("ExprType")
Numeric = int | float | complex


class Parameter(Generic[ExprType]):
    """
    A class (former `FeatureParameter`) to handle symbolic data that the user needs
    to provide its value.
    """

    pass


class Variable(Generic[ExprType]):
    """
    A class (former `VariationalParameter`) to handle variable symbolic data for
    machine learning purposes and replacing symbolic values during backend runtime
    execution.
    """

    pass


class Instruction:
    """
    Instruction class creates a named instruction with a tuple of numbers,
    parameters (`generics.Parameter`), or variables (`generics.Variable`).

    It represents expressions in terms of expanded standard instructions converted
    by the compiler library. Standard expressions are instructions that backends
    know how to interpret (defined in `<backend>.functions` module).
    """

    def __init__(self, name: str = "", *args: Any):
        self.name = name
        self.args = args

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(str(k) for k in self.args)})"


class InstructionSet(Protocol):
    """
    A group of instructions (`generics.Instruction`) that will provide the ordered
    sequence to be read and assigned to the correct module in the backend.
    """

    def __iter__(self) -> Iterable: ...
