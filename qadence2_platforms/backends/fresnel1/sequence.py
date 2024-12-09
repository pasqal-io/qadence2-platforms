from __future__ import annotations

from dataclasses import replace
from functools import reduce
from typing import Any, Callable, Optional

import numpy as np
from pulser.channels import DMM
from pulser.devices._devices import AnalogDevice
from pulser.parametrized.variable import VariableItem
from pulser.register.register_layout import RegisterLayout
from pulser.sequence.sequence import Sequence
from qadence2_ir.types import Alloc, Assign, Call, Load, Model, QuInstruct

from . import functions as add_pulse
from .device_settings import Fresnel1Settings
from .functions import PULSE_FN_MAP


class NamedPulse:
    def __init__(self, name: str, *args: Any) -> None:
        self.name = name
        self.args = args


# Fresnel-1 device (AnalogDevice with Fresnel-1 specs)
Fresnel1 = Fresnel1Settings.device


def from_model(model: Model, register: RegisterLayout) -> Sequence:
    seq = Sequence(register, Fresnel1)  # type: ignore
    seq.declare_channel("global", "rydberg_global")

    if model.directives.get("local_targets"):
        if model.directives.get("local_shifts"):
            targets = zip(
                model.directives["local_targets"],
                model.directives["local_shifts"],
            )
            detuning_map = register.define_detuning_map(
                {elem[0]: elem[1] / (2 * np.pi) for elem in targets}
            )
        else:
            targets = zip(
                model.directives["local_targets"],
                [1] * len(model.directives["local_targets"]),
            )
            detuning_map = register.define_detuning_map(dict(targets))

        seq.config_detuning_map(detuning_map, "dmm_0")

    pulses = from_instructions(seq, model.inputs, model.instructions)

    for pulse in pulses:
        fn: Optional[Callable] = getattr(
            add_pulse, PULSE_FN_MAP.get(pulse.name) or pulse.name, None
        )
        if fn is not None:
            fn(seq, *pulse.args)
        else:
            raise ValueError(f"current backend does not have pulse '{pulse.name}' implemented.")

    return seq


def from_instructions(
    sequence: Sequence,
    inputs: dict[str, Alloc],
    instructions: list[Assign | QuInstruct],
) -> list[NamedPulse]:
    variables = dict()
    temp_vars: dict[str, Any] = dict()

    for var in inputs:
        # inputs[var].size holds the points to interpolate time-dependent functions
        if inputs[var].size > 1:
            raise TypeError("This platform cannot handle time modulated variables.")

        variables[var] = sequence.declare_variable(var)

    pulses = []
    for instruction in instructions:
        if isinstance(instruction, Assign):
            assign(instruction, temp_vars, variables)

        if isinstance(instruction, QuInstruct):
            args = (
                (
                    (temp_vars.get(arg.variable) or variables[arg.variable])
                    if isinstance(arg, Load)
                    else arg
                )
                for arg in instruction.args
            )
            pulses.append(NamedPulse(instruction.name, *args))

    return pulses


def assign(
    instruction: Assign, temp_vars: dict[str, Any], variables: dict[str, VariableItem]
) -> None:
    var_name = instruction.variable

    if var_name in temp_vars or var_name in variables:
        return

    fn = instruction.value
    if isinstance(fn, Call):
        args = (
            (
                (temp_vars.get(arg.variable) or variables.get(arg.variable))
                if isinstance(arg, Load)
                else arg
            )
            for arg in fn.args
        )
        temp_vars[var_name] = _compute(fn.identifier, *args)


def _compute(fn: str, *args: Any) -> Any:
    match fn:
        case "add":
            return reduce(lambda a, b: a + b, args)
        case "mul":
            return reduce(lambda a, b: a * b, args)
        case "pow":
            return reduce(lambda a, b: a**b, args)
        case _:
            return getattr(np, fn)(*args)
