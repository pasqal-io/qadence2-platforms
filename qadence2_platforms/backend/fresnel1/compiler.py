from __future__ import annotations

from qadence2_ir.types import Model
from .interface import Interface
from . import sequence, register


def modelc(model: Model) -> Interface:
    reg = register.from_model(model)
    seq = sequence.from_model(model, reg)
    non_trainable_parameters = {k: v.size for k, v in model.inputs.items()}
    return Interface(seq, non_trainable_parameters)
