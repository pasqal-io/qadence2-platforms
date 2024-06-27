"""
This file is the API-like interface between the compilation and runtime processes
and the de facto backend library. It may be more expressive in cases where the
backend library does not provide enough customization or data handling from the
qadence-core (compilation- and runtime-wise) perspective, which is the case for
this backend.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Generic, Any, Union, Callable

import numpy as np
from pulser.channels import DMM as PulserDMM
from pulser.devices import (
    AnalogDevice as PulserAnalogDevice,
)
from pulser.devices import (
    DigitalAnalogDevice as PulserDigitalAnalogDevice,
)
from pulser.devices import (
    IroiseMVP,
)
from pulser.devices._device_datacls import BaseDevice as PulserBaseDevice
from pulser.register.base_register import BaseRegister
from pulser.register.special_layouts import (
    SquareLatticeLayout,
    TriangularLatticeLayout,
)

from qadence2_platforms.qadence_ir import Model
from qadence2_platforms.types import UserInputType, BytecodeInstructType

_dmm = PulserDMM(
    # from Pulser tutorials/dmm.html#DMM-Channel-and-Device
    clock_period=4,
    min_duration=16,
    max_duration=2**26,
    mod_bandwidth=8,
    bottom_detuning=-2 * np.pi * 20,  # detuning between 0 and -20 MHz
    total_bottom_detuning=-2 * np.pi * 2000,  # total detuning
)


# list of devices
AnalogDevice = PulserAnalogDevice
DigitalAnalogDevice = PulserDigitalAnalogDevice
Iroise = IroiseMVP
Fresnel = replace(PulserAnalogDevice.to_virtual(), dmm_objects=(_dmm,))
FresnelEOM = Fresnel


def get_backend_register(model: Model, device: PulserBaseDevice) -> BaseRegister:
    coords = np.array(model.register.qubit_positions, dtype="float64")
    max_amp = device.channels["rydberg_global"].max_amp or 5 * np.pi
    spacing_unit = device.rydberg_blockade_radius(max_amp)

    if model.directives.get("enable_digital_analog"):
        if model.register.grid_scale:
            warnings.warn(
                "This device uses atomic distance-based strategies to perform"
                " digital operations. When the `enable_digital_analog` directive is active,"
                " the option `grid_scale` is ignored. Please be aware that turning off the "
                "directive `enable_digital_analog` may result in unexpected behaviours from"
                " digital operations.",
                SyntaxWarning,
                stacklevel=2,
            )

        spacing_unit *= 1.2

    spacing_unit *= model.register.grid_scale
    coords *= spacing_unit

    if model.register.grid_type == "triangular":
        layout = TriangularLatticeLayout(n_traps=61, spacing=spacing_unit)
        transform = np.array([[1.0, 0.0], [0.5, 0.8660254037844386]])
        coords @= transform
    else:
        layout = SquareLatticeLayout(7, 7, spacing=spacing_unit)

    traps = layout.get_traps_from_coordinates(*coords)
    register = layout.define_register(*traps, qubit_ids=range(len(traps)))

    return register


class BackendInstructResult(Generic[UserInputType, BytecodeInstructType]):
    def __init__(self, *args: Any, fn: Union[Callable, None] = None):
        if fn is None:
            raise ValueError("Must declare `fn` argument.")
        self._fn = fn
        self._args = args

    @property
    def fn(self) -> Callable:
        return self._fn

    @property
    def args(self) -> tuple[Any, ...]:
        return self._args


SequenceType = list[Callable[..., Any]]
