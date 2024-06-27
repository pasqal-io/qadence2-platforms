from __future__ import annotations

from .register import RegisterInterface
from qadence2_platforms.qadence_ir import Model


def get_backend_register(model: Model, _device: None = None) -> RegisterInterface:
    return RegisterInterface(model.register)
