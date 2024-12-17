from __future__ import annotations

from typing import Any

import numpy as np
from pulser import Pulse
from pulser.parametrized.variable import VariableItem, Variable
from pulser.sequence import Sequence
from pulser.waveforms import ConstantWaveform, BlackmanWaveform, RampWaveform, CompositeWaveform

from qadence2_platforms.backends._base_analog.functions import (
    Direction,
    Duration,
    base_parse_native_observables,
    BaseQuTiPObservablesParser,
    DEFAULT_AMPLITUDE,
    DEFAULT_DETUNING,
)

# TODO: re-introduce `Support` to account for "local" and "global" on the `channel` arg

# pulse function mapping.
# keys are the `qadence2_ir.Model` standard, values are the function names below
PULSE_FN_MAP = {
    "rx": "rx",
    "ry": "ry",
    "not": "x",
    "h": "h",
}

# reference to function to parse native observables on `Interface`'s `expectation` method
parse_native_observables = base_parse_native_observables
# qutip observable parser
QuTiPObservablesParser = BaseQuTiPObservablesParser


def dyn_pulse(
    sequence: Sequence,
    duration: VariableItem | float,
    amplitude: VariableItem | float,
    detuning: VariableItem | float,
    phase: VariableItem | float,
    **_: Any,
) -> None:
    """
    Dynamic pulse to simulate a specific time-dependent hamiltonian for neutral-atom devices.

    Args:
        sequence: a `pulser.sequence.Sequence` instance
        duration: duration of the pulse in dimensionless units
        amplitude: amplitude of the pulse in dimensionless units
        detuning: detuning of the pulse in dimensionless units
        phase: phase in radians
    """
    max_amp = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    max_abs_detuning = (
        sequence.device.channels["rydberg_global"].max_abs_detuning or DEFAULT_DETUNING
    )

    # FIXME: Centralize unit converions
    duration *= 1000 * 2 * np.pi / max_amp  # type: ignore
    amplitude *= max_amp  # type: ignore
    detuning *= max_abs_detuning  # type: ignore

    new_amplitude = ConstantWaveform(duration, amplitude)
    new_detuning = ConstantWaveform(duration, detuning)

    p = Pulse(new_amplitude, new_detuning, phase)
    sequence.add(p, channel="global")


def piecewise_pulse(
    sequence: Sequence,
    duration: Variable | VariableItem,
    amplitude: Variable,
    detuning: Variable,
    phase: VariableItem | float,
    **_: Any,
) -> None:
    """
    Dynamic pulse to simulate a specific piecewise time-dependent hamiltonian for neutral-atom devices.

    Args:
        sequence: a `pulser.sequence.Sequence` instance
        duration: duration of the pulse in dimensionless units
        amplitude: amplitude of the pulse in dimensionless units
        detuning: detuning of the pulse in dimensionless units
        phase: phase in radians
    """
    max_amp = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    max_abs_detuning = (
        sequence.device.channels["rydberg_global"].max_abs_detuning or DEFAULT_DETUNING
    )

    dur_factor = 1000 * 2 * np.pi / max_amp  # type: ignore
    amp_factor = max_amp  # type: ignore
    det_factor = max_abs_detuning  # type: ignore

    # Needed so a size = 1 variable is made iterable
    duration = [duration] if isinstance(duration, VariableItem) else duration

    for i, dur in enumerate(duration):
        amp_wf = RampWaveform(
            dur * dur_factor, amplitude[i] * amp_factor, amplitude[i + 1] * amp_factor
        )
        det_wf = RampWaveform(
            dur * dur_factor, detuning[i] * det_factor, detuning[i + 1] * det_factor
        )
        sequence.add(Pulse(amp_wf, det_wf, phase), "global")


def rx(
    sequence: Sequence,
    angle: VariableItem | float,
    **_: Any,
) -> None:
    rotation(sequence, angle, Direction.X)


def ry(
    sequence: Sequence,
    angle: VariableItem | float,
    **_: Any,
) -> None:
    rotation(sequence, angle, Direction.Y)


def x(sequence: Sequence, **_: Any) -> None:
    rotation(sequence, angle=np.pi, direction=Direction.X)


def h(
    sequence: Sequence,
    duration: VariableItem | float = np.pi,
    support_list: str = "global",
    **_: Any,
) -> None:
    amplitude = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    duration *= 1000 * 2 * np.pi / amplitude

    pi2_wf = BlackmanWaveform(1000, np.pi / 2)
    sequence.add(
        Pulse.ConstantDetuning(pi2_wf, 0, np.pi / 2, post_phase_shift=np.pi), channel="global"
    )


def rotation(
    sequence: Sequence,
    angle: VariableItem | float,
    direction: Direction | float,
    **_: Any,
) -> None:

    amplitude = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    duration = 1000 * angle / amplitude
    detuning = 0

    phase: Any
    match direction:
        case Direction.X:
            phase = 0
        case Direction.Y:
            phase = np.pi / 2
        case _:
            phase = direction

    sequence.add(Pulse.ConstantPulse(duration, amplitude, detuning, phase), "global")


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
    duration: VariableItem | float | Duration,
    detuning: VariableItem | float,
    concurrent: bool = False,
    **_: Any,
) -> None:
    max_amp = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    time_scale = 1000 * 2 * np.pi / max_amp
    local_pulse_core(sequence, duration, time_scale, detuning, concurrent)


def local_pulse_core(
    sequence: Sequence,
    duration: VariableItem | float | Duration,
    time_scale: float,
    detuning: VariableItem | float,
    concurrent: bool = False,
    **kwargs: Any,
) -> None:
    max_abs_detuning = (
        sequence.device.channels["rydberg_global"].max_abs_detuning or DEFAULT_DETUNING
    )

    if duration == Duration.FILL:
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
