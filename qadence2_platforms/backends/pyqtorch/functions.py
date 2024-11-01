from __future__ import annotations

from typing import cast, Iterable

import pyqtorch as pyq
from pyqtorch.hamiltonians import Observable
from pyqtorch.primitives import Primitive

from qadence2_platforms.backends.utils import InputType, Support


def _get_op(op: InputType) -> Primitive | None:
    if op.is_symbol is True:
        return getattr(pyq, op.head.upper(), None)

    if op.is_quantum_operator is True:
        op_args_item: InputType = op.args[0].args
        return getattr(pyq, op_args_item[0].upper(), None)

    return _get_op(op.args[0])


def _get_native_op(op: InputType) -> Primitive:
    native_op = _get_op(op)
    if native_op is not None:
        support: Support = cast(Support, op.args[1])
        return native_op(*support.target, support.control)

    raise ValueError(f"Observable {op} not found")


def _iterate_over_obs(op: Iterable | InputType) -> list[Primitive]:
    if isinstance(op, Iterable):
        return [_get_native_op(arg) for arg in op]
    return [_get_native_op(arg) for arg in op.args]


def _is_arith_expr(expr: InputType) -> bool:
    return (
        getattr(expr, "is_addition", None)
        or getattr(expr, "is_multiplication", None)
        or getattr(expr, "is_kronecker_product", None)
        or getattr(expr, "is_power", None)
    )


def load_observables(observable: list[InputType] | InputType) -> Observable:
    pyq_observables: list[Primitive] | Primitive
    if isinstance(observable, list) or _is_arith_expr(observable) is True:
        pyq_observables = _iterate_over_obs(observable)
        # return Observable(pyq_observables)
    else:
        pyq_observables = _get_native_op(observable)
    #
    # if _is_arith_expr(observable) is True:
    #     return Observable(_iterate_over_obs(observable))

    return Observable(pyq_observables)
