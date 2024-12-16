from __future__ import annotations

from qadence2_ir.types import Model

from . import register, sequence
from .interface import Interface


def compile_to_backend(model: Model) -> Interface:
    reg = register.from_model(model)
    seq = sequence.from_model(model, reg)
    non_trainable_parameters = {k for k, v in model.inputs.items() if not v.is_trainable}
    return Interface(seq, non_trainable_parameters)
