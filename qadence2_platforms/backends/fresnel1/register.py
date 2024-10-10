from __future__ import annotations

import warnings

import numpy as np
from pulser.devices._devices import AnalogDevice
from pulser.register.register_layout import RegisterLayout
from qadence2_ir.types import Model

warnings.filterwarnings("ignore", category=UserWarning)
warnings.formatwarning = lambda msg, *args, **kwargs: f"WARNING: {msg}\n"


def from_model(model: Model) -> RegisterLayout:
    model_register = model.register

    if model_register.num_qubits < 1:
        raise ValueError("No qubit available in the register.")

    if model_register.grid_type and model_register.grid_type != "triangular":
        warnings.warn(
            "Fresnel-1 only supoorts triangular grids at the moment.",
            SyntaxWarning,
            stacklevel=2,
        )

    if model_register.grid_scale != 1.0:
        warnings.warn(
            "Currently, Fresnel-1 uses a fixed grid.",
            SyntaxWarning,
            stacklevel=2,
        )

    if model.directives.get("enable_digital_analog"):
        warnings.warn(
            "Fresnel-1 uses a fixed grid and does not have digital channels.\n"
            + "Digital operations using atomic distance-based strategies\n"
            + "may not behave as expected.",
            SyntaxWarning,
            stacklevel=2,
        )

    coords = model_register.qubit_positions

    if not coords:
        shift = model_register.num_qubits // 2
        coords = [(p - shift, 0) for p in range(model_register.num_qubits)]

    transform = np.array([[1.0, 0.0], [0.5, 0.8660254037844386]])
    coords = coords @ transform

    layout = AnalogDevice.calibrated_register_layouts["TriangularLatticeLayout(61, 5.0µm)"]
    coords *= 5
    traps = layout.get_traps_from_coordinates(*coords)
    register = layout.define_register(*traps, qubit_ids=range(len(traps)))

    return register  # type: ignore
