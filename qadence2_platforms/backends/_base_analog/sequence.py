from __future__ import annotations

from functools import reduce
from typing import Any

import numpy as np
from pulser.parametrized.variable import VariableItem
from pulser.sequence.sequence import Sequence
from qadence2_ir.types import Alloc, Assign, Call, Load, QuInstruct


class NamedPulse:
    def __init__(self, name: str, *args: Any) -> None:
        self.name = name
        self.args = args


def from_instructions(
    sequence: Sequence,
    inputs: dict[str, Alloc],
    instructions: list[Assign | QuInstruct],
    allow_time_dependent: bool = False,
) -> list[NamedPulse]:
    variables = dict()
    temp_vars: dict[str, Any] = dict()

    for var in inputs:
        # inputs[var].size holds the points to interpolate time-dependent functions
        if inputs[var].size > 1:
            if allow_time_dependent:
                variables[var] = sequence.declare_variable(var, size=inputs[var].size)
            else:
                raise TypeError("This platform cannot handle time modulated variables.")
        else:
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
