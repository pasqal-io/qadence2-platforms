from __future__ import annotations

from typing import Iterable, cast

import pyqtorch as pyq
from pyqtorch.hamiltonians import Observable
from pyqtorch.primitives import Primitive

from qadence2_platforms.backends.utils import InputType, Support


def parse_native_observables(observable: list[InputType] | InputType) -> Observable:
    return PyQObservablesParser.build(observable)


class PyQObservablesParser:
    @classmethod
    def _get_op(cls, op: InputType) -> Primitive | None:
        if op.is_symbol is True:
            symbol: str = cast(str, op.head)
            return getattr(pyq, symbol.upper(), None)

        arg: InputType = cast(InputType, op.args[0])

        if op.is_quantum_operator is True:
            op_args_item: InputType = cast(InputType, arg.args)
            return getattr(pyq, op_args_item[0].upper(), None)

        return cls._get_op(arg)

    @classmethod
    def _get_native_op(cls, op: InputType) -> Primitive:
        native_op = cls._get_op(op)

        if native_op is not None:
            support: Support = cast(Support, op.args[1])
            return native_op(*support.target, support.control)

        raise ValueError(f"Observable {op} not found")

    @classmethod
    def _iterate_over_obs(cls, op: Iterable | InputType) -> list[Primitive]:
        if isinstance(op, Iterable):
            return [cls._get_native_op(arg) for arg in op]

        return [cls._get_native_op(arg) for arg in op.args]  # type: ignore [arg-type, union-attr]

    @classmethod
    def _is_arith_expr(cls, expr: InputType) -> bool | None:
        return (
            getattr(expr, "is_addition", None)
            or getattr(expr, "is_multiplication", None)
            or getattr(expr, "is_kronecker_product", None)
            or getattr(expr, "is_power", None)
        )

    @classmethod
    def build(cls, observable: list[InputType] | InputType) -> Observable:
        pyq_observables: list[Primitive] | Primitive

        if isinstance(observable, list) or cls._is_arith_expr(observable) is True:
            pyq_observables = cls._iterate_over_obs(observable)

        else:
            pyq_observables = cls._get_native_op(observable)

        return Observable(pyq_observables)
