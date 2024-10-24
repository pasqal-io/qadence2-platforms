from __future__ import annotations

import pyqtorch as pyq
from pyqtorch.hamiltonians import Observable
from pyqtorch.primitives import Primitive

from qadence2_platforms.backends.pyqtorch.utils import InputType


def _get_op(op: InputType) -> Primitive:
    if op.head.is_symbol is True:
        return getattr(pyq, op.head.args[0].upper(), None)

    if op.head.is_quantum_operator is True:
        op_args = op.args[0].args
        return getattr(pyq, op_args[0].upper(), None)

    return getattr(pyq, op.args[0].upper(), None)


def _get_native_op(op: InputType) -> Primitive:
    native_op = _get_op(op)
    if native_op is not None:
        support = op.args[1]
        return native_op(*support.target, support.control)

    raise ValueError(f"Observable {op} not found")


def load_observables(observable: list[InputType] | InputType) -> Observable:
    if isinstance(observable, list):
        pyq_observables: list[Primitive] | list = []
        for op in observable:
            pyq_observables.append(_get_native_op(op))
        return Observable(*pyq_observables)

    return Observable(_get_native_op(observable))
