from __future__ import annotations

from pulser.register import RegisterLayout
from qadence2_ir.types import Model

from qadence2_platforms.backends._base_analog.register import RegisterTransform


def from_model(model: Model) -> RegisterLayout:
    model_register = model.register

    coords = model_register.qubit_positions
