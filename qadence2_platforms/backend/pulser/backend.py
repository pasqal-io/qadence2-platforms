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
from functools import lru_cache, partial
from typing import Any, Iterator, Optional, cast

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
from pulser.sequence import Sequence as PulserSequence

from qadence2_platforms.backend.pulser import EmbeddingModule
from qadence2_platforms.qadence_ir import Model

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


class InstructPartialResult:
    def __init__(self, fn: partial, params: Any):
        if fn is None:
            raise ValueError("Must declare `fn` argument.")
        self.fn: partial = fn
        self._args: tuple[Any, ...] = params if isinstance(params, tuple) else (params,)

    @property
    def args(self) -> tuple[Any, ...]:
        return self._args

    def __iter__(self) -> Iterator[Any]:
        return iter((self.fn, self.args))

    def __len__(self) -> int:
        return len(self.args)

    def __repr__(self) -> str:
        return (
            f"InstructPartialResult(fn={self.fn.func.__name__},"
            f" params=[{' '.join(p.variable for p in self.args)}])"
        )


class BackendPartialSequence:
    def __init__(self, *instructions: Any):
        self.partial_instr: tuple[InstructPartialResult, ...] = instructions

    @staticmethod
    @lru_cache
    def get_fn_args(fn: partial) -> Any:
        match fn.func.__name__:
            case "rotation":
                return ["angle", "direction"]
            case "pulse":
                return ["duration", "amplitude", "detuning", "phase"]
            case "free_evolution":
                return ["duration"]
            case _:
                return []

    def evaluate(
        self, embedding: EmbeddingModule, values: Optional[dict] = None
    ) -> PulserSequence:
        seq: Optional[PulserSequence] = None
        assigned_values: dict = embedding(values)
        for fn, params in self.partial_instr:
            resolved_params: tuple[Any, ...] = ()
            params = cast(tuple, params)
            fn = cast(partial, fn)
            for param in params:
                resolved_params += (assigned_values[param.variable],)
            seq = fn(**dict(zip(self.get_fn_args(fn), resolved_params)))

        if seq:
            return seq
        raise ValueError("pulser sequence must not be None.")
