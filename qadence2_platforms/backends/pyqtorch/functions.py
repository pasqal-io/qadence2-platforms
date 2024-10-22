from __future__ import annotations

import pyqtorch as pyq
from pyqtorch.hamiltonians import Observable
from pyqtorch.primitives import Primitive

from qadence2_platforms.backends.pyqtorch.utils import InputType


def _get_op(op: InputType) -> Primitive:
    return getattr(pyq, op.head.upper(), None)


def load_observables(observable: list[InputType] | InputType) -> Observable:
    if isinstance(observable, list):
        pyq_observables: list[Primitive] | list = []
        for op in observable:
            native_op = _get_op(op)
            if native_op is not None:
                pyq_observables.append(native_op)
            else:
                raise ValueError(f"Observable {op} not found")
        return Observable(*pyq_observables)
    return Observable(_get_op(observable))
