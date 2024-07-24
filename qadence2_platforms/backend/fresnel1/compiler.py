from __future__ import annotations

from qadence2_ir.types import Model
from .interface import Interface
from . import sequence, register


def modelc(model: Model) -> Interface:
    reg = register.from_model(model)
    seq = sequence.from_model(model, reg)
    return Interface(seq)
