from __future__ import annotations

import warnings

from pulser.register.register_layout import RegisterLayout
from qadence2_ir.types import Model

from .device_settings import Fresnel1Settings
from .._base_analog.register import RegisterTransform, RegisterResolver

warnings.filterwarnings("ignore", category=UserWarning)
warnings.formatwarning = lambda msg, *args, **kwargs: f"WARNING: {msg}\n"


def from_model(model: Model) -> RegisterLayout:
    register = RegisterResolver.resolve_from_model(
        model=model,
        device_settings=Fresnel1Settings,
        grid_scale_fn=check_grid_scale,
        grid_type_fn=check_grid_type,
        directives_fn=check_directives,
        register_transform_fn=from_layout,
    )

    return register


def from_layout(register_transform: RegisterTransform) -> RegisterLayout:
    """
    Transforms a data from coordinates into a register with calibrated layout.

    Args:
        register_transform (RegisterTransform): The instance of RegisterTransform containing
            the coordinates and device settings

    Returns:
        A register data with calibrated layout
    """

    layout = register_transform.get_calibrated_layout("TriangularLatticeLayout(61, 5.0µm)")
    traps = layout.get_traps_from_coordinates(*register_transform.coords)
    register = layout.define_register(*traps, qubit_ids=range(len(traps)))

    return register  # type: ignore


def check_grid_scale(model: Model) -> None:
    if Fresnel1Settings.scale_in_range(model.register.grid_scale):
        warnings.warn(
            "Currently, Fresnel-1 uses a fixed grid.",
            SyntaxWarning,
            stacklevel=2,
        )


def check_grid_type(model: Model) -> None:
    if (
        model.register.grid_type
        and model.register.grid_type not in Fresnel1Settings.available_grid_types
    ):
        warnings.warn(
            "Fresnel-1 only supports triangular grids at the moment.",
            SyntaxWarning,
            stacklevel=2,
        )
        model.register.grid_type = Fresnel1Settings.available_grid_types[0]


def check_directives(model: Model) -> None:
    if model.directives.get("enable_digital_analog"):
        warnings.warn(
            "Fresnel-1 uses a fixed grid and does not have digital channels.\n"
            + "Digital operations using atomic distance-based strategies\n"
            + "may not behave as expected.",
            SyntaxWarning,
            stacklevel=2,
        )
