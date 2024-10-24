from __future__ import annotations

from typing import cast

import pyqtorch as pyq
from pyqtorch.hamiltonians import Observable
from pyqtorch.primitives import Primitive

from qadence2_platforms.backends.pyqtorch.utils import InputType, Support


def _get_op(op: InputType) -> Primitive | None:
    if op.is_symbol is True:
        return getattr(pyq, op.head.upper(), None)

    if op.is_quantum_operator is True:
        op_args_item: InputType = op.args[0].args
        return getattr(pyq, op_args_item[0].upper(), None)

    op_args: str = cast(str, op.args[0])
    return getattr(pyq, op_args.upper(), None)


def _get_native_op(op: InputType) -> Primitive:
    native_op = _get_op(op)
    if native_op is not None:
        support: Support = cast(Support, op.args[1])
        return native_op(*support.target, support.control)

    raise ValueError(f"Observable {op} not found")


def load_observables(observable: list[InputType] | InputType) -> Observable:
    if isinstance(observable, list):
        pyq_observables: list[Primitive] | list = []
        for op in observable:
            pyq_observables.append(_get_native_op(op))
        return Observable(*pyq_observables)

    return Observable(_get_native_op(observable))
