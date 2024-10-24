from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pulser.parametrized.variable import VariableItem
from pulser.pulse import Pulse
from pulser.sequence.sequence import Sequence
from pulser.waveforms import ConstantWaveform

DEFAULT_AMPLITUDE = 4 * np.pi
DEFAULT_DETUNING = 10 * np.pi


# pulse function mapping.
# keys are the `qadence2_ir.Model` standard, values are the function names below
PULSE_FN_MAP = {
    "rx": "rx",
    "ry": "ry",
    "not": "x",
    "h": "h",
}


def dyn_pulse(
    sequence: Sequence,
    duration: VariableItem | float,
    amplitude: VariableItem | float,
    detuning: VariableItem | float,
    phase: VariableItem | float,
    **_: Any,
) -> None:
    max_amp = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    max_abs_detuning = (
        sequence.device.channels["rydberg_global"].max_abs_detuning or DEFAULT_DETUNING
    )

    duration *= 1000 * 2 * np.pi / max_amp  # type: ignore
    amplitude *= max_amp  # type: ignore
    detuning *= max_abs_detuning  # type: ignore
    pulse = Pulse.ConstantPulse(duration, amplitude, detuning, phase)
    sequence.add(pulse, "global")  # type: ignore


def rx(
    sequence: Sequence,
    angle: VariableItem | float,
    **_: Any,
) -> None:
    rotation(sequence, angle, "x")


def ry(
    sequence: Sequence,
    angle: VariableItem | float,
    **_: Any,
) -> None:
    rotation(sequence, angle, "y")


def x(sequence: Sequence, **_: Any) -> None:
    rotation(sequence, angle=np.pi, direction="x")


def h(
    sequence: Sequence,
    duration: VariableItem | float = 1000.0,
    **_: Any,
) -> None:
    support_list = "GLOBAL"
    amplitude = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    duration *= 1000 * 2 * np.pi / amplitude
    duration = int(duration) if duration > 16 else 16

    pulse = Pulse.ConstantPulse(duration, amplitude, support_list)
    sequence.add(pulse, "global")


def rotation(
    sequence: Sequence,
    angle: VariableItem | float,
    direction: Literal["x", "y"] | float,
    **_: Any,
) -> None:

    amplitude = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    duration = 1000 * angle / amplitude
    detuning = 0

    phase: Any
    match direction:
        case "x":
            phase = 0
        case "y":
            phase = np.pi / 2
        case _:
            phase = direction

    pulse = Pulse.ConstantPulse(duration, amplitude, detuning, phase)
    sequence.add(pulse, "global")  # type: ignore


def dyn_wait(
    sequence: Sequence,
    duration: VariableItem | float,
    *args: Any,
    **kwargs: Any,
) -> None:
    max_amp = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE

    duration *= 1000 * 2 * np.pi / max_amp  # type: ignore

    sequence.delay(int(duration), "global")  # type: ignore


def apply_local_shifts(sequence: Sequence, **_: Any) -> None:
    max_abs_detuning = (
        sequence.device.channels["rydberg_global"].max_abs_detuning or DEFAULT_DETUNING
    )
    time_scale = 1000 * 2 * np.pi / max_abs_detuning
    local_pulse_core(sequence, duration=1.0, time_scale=time_scale, detuning=1.0, concurrent=False)


def local_pulse(
    sequence: Sequence,
    duration: VariableItem | float | Literal["fill"],
    detuning: VariableItem | float,
    concurrent: bool = False,
    **_: Any,
) -> None:
    max_amp = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    time_scale = 1000 * 2 * np.pi / max_amp
    local_pulse_core(sequence, duration, time_scale, detuning, concurrent)


def local_pulse_core(
    sequence: Sequence,
    duration: VariableItem | float | Literal["fill"],
    time_scale: float,
    detuning: VariableItem | float,
    concurrent: bool = False,
    **kwargs: Any,
) -> None:
    max_abs_detuning = (
        sequence.device.channels["rydberg_global"].max_abs_detuning or DEFAULT_DETUNING
    )

    if duration == "fill":
        if not concurrent:
            raise SyntaxError("The option `fill` can only be used on the `concurrent` mode")

        duration = sequence.get_duration("global") - sequence.get_duration("dmm_0")
    else:
        duration *= time_scale  # type: ignore

    detuning *= -1 * max_abs_detuning  # type: ignore

    sequence.add_dmm_detuning(
        ConstantWaveform(duration, detuning),
        "dmm_0",
        "no-delay" if concurrent else "min-delay",
    )
