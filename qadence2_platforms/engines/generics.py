from __future__ import annotations
from typing import Protocol, TypeVar


ExprType = TypeVar("ExprType")


class CompiledExpr(Protocol):
    """
    Protocol class to be used to represent symbolic, numeric, numerical or expression classes
    that were transformed by the compiler library. Use it when context requires expressions
    instead of importing it directly from the compiler.
    """

    def __mul__(self, other) -> ExprType: ...
    def __rmul__(self, other) -> ExprType: ...
    def __pow__(self, other) -> ExprType: ...
    def __rpow__(self, other) -> ExprType: ...
    def __truediv__(self, other) -> ExprType: ...
    def __rtruediv__(self, other) -> ExprType: ...
    def __add__(self, other) -> ExprType: ...
    def __radd__(self, other) -> ExprType: ...
    def __neg__(self) -> ExprType: ...
    def __sub__(self, other) -> ExprType:...
    def __rsub__(self, other) -> ExprType: ...

