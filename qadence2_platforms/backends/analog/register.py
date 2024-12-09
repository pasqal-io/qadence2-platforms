from __future__ import annotations

import warnings
from typing import Any

from pulser.register import RegisterLayout, Register
from qadence2_ir.types import Model

from qadence2_platforms.backends._base_analog.register import RegisterTransform, RegisterResolver
from qadence2_platforms.backends.analog.device_settings import AnalogSettings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.formatwarning = lambda msg, *args, **kwargs: f"WARNING: {msg}\n"


def from_model(model: Model) -> RegisterLayout:
    register = RegisterResolver.resolve_from_model(
        model=model,
        device_settings=AnalogSettings,
        grid_scale_fn=check_grid_scale,
        grid_type_fn=check_grid_type,
        directives_fn=check_directives,
        register_transform_fn=from_coords,
    )

    return register


def from_coords(register_transform: RegisterTransform) -> RegisterLayout:
    register = Register.from_coordinates(register_transform.coords)
    return register  # type: ignore


def check_grid_scale(grid_scale: float) -> None:
    pass


def check_grid_type(grid_type: float) -> None:
    pass


def check_directives(directives: dict[str, Any]) -> None:
    pass
