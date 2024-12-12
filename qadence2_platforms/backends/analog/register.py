from __future__ import annotations

import warnings

from pulser.register import RegisterLayout, Register
from qadence2_ir.types import Model

from qadence2_platforms.backends._base_analog.register import RegisterTransform, RegisterResolver
from qadence2_platforms.backends.analog.device_settings import AnalogSettings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.formatwarning = lambda msg, *args, **kwargs: f"WARNING: {msg}\n"


def from_model(model: Model) -> RegisterLayout:
    """
    Gets information from IR model data to generate a register/register layout.

    Args:
        model (Model): IR model data

    Returns:
        `RegisterLayout` data
    """

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
    """
    Function to transform the coordinates into appropriate register layout.

    Args:
        register_transform (RegisterTransform): Register transform instance containing
            the coordinates and register-related data

    Returns:
        `RegisterLayout` data
    """
    register = Register.from_coordinates(register_transform.coords)
    return register  # type: ignore


def check_grid_scale(model: Model) -> None:
    pass


def check_grid_type(model: Model) -> None:
    pass


def check_directives(model: Model) -> None:
    pass
