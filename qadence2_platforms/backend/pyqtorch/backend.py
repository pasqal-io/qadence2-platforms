from __future__ import annotations

from qadence2_ir import Model

from .register import RegisterInterface


def get_backend_register(model: Model, _device: None = None) -> RegisterInterface:
    return RegisterInterface(model.register)
