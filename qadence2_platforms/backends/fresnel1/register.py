from __future__ import annotations

import warnings

import numpy as np
from pulser.devices._devices import AnalogDevice
from pulser.register.register_layout import RegisterLayout
from qadence2_ir.types import Model

from .device_settings import Fresnel1Settings
from .._base_analog.register import RegisterTransform

warnings.filterwarnings("ignore", category=UserWarning)
warnings.formatwarning = lambda msg, *args, **kwargs: f"WARNING: {msg}\n"


def from_model(model: Model) -> RegisterLayout:
    model_register = model.register

    if model_register.num_qubits < 1:
        raise ValueError("No qubit available in the register.")

    # # if model_register.grid_type and model_register.grid_type != "triangular":
    # if model_register.grid_type and model_register.grid_type != Fresnel1Settings.grid_type:
    #     warnings.warn(
    #         "Fresnel-1 only supoorts triangular grids at the moment.",
    #         SyntaxWarning,
    #         stacklevel=2,
    #     )
    Fresnel1Settings.check_grid_type(model_register.grid_type)

    # # if model_register.grid_scale != 1.0:
    # if Fresnel1Settings.scale_in_range(model_register.grid_scale):
    #     warnings.warn(
    #         "Currently, Fresnel-1 uses a fixed grid.",
    #         SyntaxWarning,
    #         stacklevel=2,
    #     )
    Fresnel1Settings.check_grid_scale(model_register.grid_scale)

    # if model.directives.get("enable_digital_analog"):
    #     warnings.warn(
    #         "Fresnel-1 uses a fixed grid and does not have digital channels.\n"
    #         + "Digital operations using atomic distance-based strategies\n"
    #         + "may not behave as expected.",
    #         SyntaxWarning,
    #         stacklevel=2,
    #     )
    Fresnel1Settings.check_directives(model.directives)

    # coords = model_register.qubit_positions
    #
    # if not coords:
    #     shift = model_register.num_qubits // 2
    #     coords = [(p - shift, 0) for p in range(model_register.num_qubits)]
    #
    # transform = np.array([[1.0, 0.0], [0.5, 0.8660254037844386]])
    # coords = coords @ transform
    #
    # layout = AnalogDevice.calibrated_register_layouts["TriangularLatticeLayout(61, 5.0µm)"]
    #
    # coords *= 5
    # traps = layout.get_traps_from_coordinates(*coords)
    # register = layout.define_register(*traps, qubit_ids=range(len(traps)))

    register_transform = RegisterTransform(
        grid_transform="triangular",
        grid_scale=model_register.grid_scale,
        coords=model_register.qubit_positions,
        num_qubits=model_register.num_qubits,
        device_settings=Fresnel1Settings,
    )

    layout = register_transform.get_calibrated_layout("TriangularLatticeLayout(61, 5.0µm)")

    traps = layout.get_traps_from_coordinates(*(register_transform.coords * 5))
    register = layout.define_register(*traps, qubit_ids=range(len(traps)))

    return register  # type: ignore
