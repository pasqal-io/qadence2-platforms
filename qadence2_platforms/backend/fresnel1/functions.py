from __future__ import annotations

from typing import Any, Literal
import numpy as np

from pulser.sequence.sequence import Sequence
from pulser.parametrized import Variable
from pulser.parametrized.variable import VariableItem
from pulser.waveforms import ConstantWaveform


DEFAULT_AMPLITUDE = 4 * np.pi
DEFAULT_DETUNING = 10 * np.pi


def get_variable(
    sequence: Sequence, var: str | float | int
) -> Variable | VariableItem | float | int:
    if not isinstance(var, str | float | int):
        raise TypeError("Only `str`, `float` and `int` are allowed as variables.")

    if isinstance(var, str):
        if var in sequence.declared_variables:
            return sequence.declared_variables[var]
        return sequence.declare_variable(var)

    return var


def pulse(
    sequence: Sequence,
    duration: str | float,
    amplitude: str | float,
    detuning: str | float,
    phase: str | float,
    **_: Any,
) -> None:
    max_amp = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    max_abs_detuning = (
        sequence.device.channels["rydberg_global"].max_abs_detuning or DEFAULT_DETUNING
    )

    _duration = get_variable(sequence, duration)
    _amplitude = get_variable(sequence, amplitude)
    _detuning = get_variable(sequence, detuning)
    _phase = get_variable(sequence, phase)

    _duration *= 1000 * 2 * np.pi / max_amp
    _amplitude *= max_amp
    _detuning *= max_abs_detuning

    sequence.enable_eom_mode("global", amp_on=_amplitude, detuning_on=_detuning)
    sequence.add_eom_pulse("global", duration=int(duration), phase=_phase)
    sequence.disable_eom_mode("global")


def rx(
    sequence: Sequence,
    angle: str | float,
    **_: Any,
) -> None:
    rotation(sequence, angle, "x")


def ry(
    sequence: Sequence,
    angle: str | float,
    **_: Any,
) -> None:
    rotation(sequence, angle, "y")


def rotation(
    sequence: Sequence,
    angle: str | float,
    direction: Literal["x", "y"] | float,
    **_: Any,
) -> None:
    _angle = get_variable(sequence, angle)

    amplitude = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    duration = 1000 * _angle / amplitude
    detuning = 0

    match direction:
        case "x":
            phase = 0
        case "y":
            phase = np.pi / 2
        case _:
            phase = direction

    sequence.enable_eom_mode("global", amp_on=amplitude, detuning_on=detuning)
    sequence.add_eom_pulse("global", duration=duration, phase=phase)  # type: ignore
    sequence.disable_eom_mode("global")


def free_evolution(
    sequence: Sequence,
    duration: str | float,
    *_args: Any,
    **_kwargs: Any,
) -> None:
    max_amp = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE

    _duration = get_variable(sequence, duration)
    _duration *= 1000 * 2 * np.pi / max_amp

    sequence.delay(int(duration), "global")  # type: ignore


def apply_local_shifts(sequence: Sequence, **_: Any) -> None:
    max_abs_detuning = (
        sequence.device.channels["rydberg_global"].max_abs_detuning or DEFAULT_DETUNING
    )
    time_scale = 1000 * 2 * np.pi / max_abs_detuning
    _local_pulse_core(
        sequence, duration=1.0, time_scale=time_scale, detuning=1.0, concurrent=False
    )


def local_pulse(
    sequence: Sequence,
    duration: str | float | Literal["fill"],
    detuning: str | float,
    concurrent: bool = False,
    **_: Any,
) -> None:
    max_amp = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    time_scale = 1000 * 2 * np.pi / max_amp
    _local_pulse_core(sequence, duration, time_scale, detuning, concurrent)


def _local_pulse_core(
    sequence: Sequence,
    duration: str | float | Literal["fill"],
    time_scale: float,
    detuning: str | float,
    concurrent: bool = False,
    **_kwargs: Any,
) -> None:
    max_abs_detuning = (
        sequence.device.channels["rydberg_global"].max_abs_detuning or DEFAULT_DETUNING
    )

    if duration == "fill":
        if not concurrent:
            raise SyntaxError(
                "The option `fill` can only be used on the `concurrent` mode"
            )

        _duration = sequence.get_duration("global") - sequence.get_duration("dmm_0")
    else:
        _duration = get_variable(sequence, duration)
        _duration *= time_scale  # type: ignore

    _detuning = get_variable(sequence, detuning)
    _detuning *= -1 * max_abs_detuning

    sequence.add_dmm_detuning(
        ConstantWaveform(_duration, _detuning),
        "dmm_0",
        "no-delay" if concurrent else "min-delay",
    )
