from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pulser.sequence import Sequence
from pulser.waveforms import ConstantWaveform

from qadence2_platforms.types import Scalar
from qadence2_platforms.qadence_ir import Support

DEFAULT_AMPLITUDE = 4 * np.pi
DEFAULT_DETUNING = 10 * np.pi


# TODO: redefine the functions according to the needs of `BackendInstruct` that is
#  dependant on `Model` data (`QuInstruct`, `Support`, `Assign`, etc)

def pulse(
    sequence: Sequence,
    support: Support,
    duration: Scalar,
    amplitude: Scalar,
    detuning: Scalar,
    phase: Scalar,
**_: Any,
    ) -> None:
    support_list = "LOCAL" if support.target else "GLOBAL"
    max_amp = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    max_abs_detuning = (
        sequence.device.channels["rydberg_global"].max_abs_detuning or DEFAULT_DETUNING
    )

    duration *= 1000 * 2 * np.pi / max_amp
    amplitude *= max_amp
    detuning *= max_abs_detuning

    sequence.enable_eom_mode(support_list, amp_on=amplitude, detuning_on=detuning)
    sequence.add_eom_pulse(support_list, duration=int(duration), phase=phase)
    sequence.disable_eom_mode(support_list)


def rotation(
    sequence: Sequence,
    support: Support,
    angle: Scalar,
    direction: Literal["x", "y"] | Scalar,
    **_: Any,
) -> None:
    support_list = "LOCAL" if support.target else "GLOBAL"
    amplitude = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    duration = 1000 * angle / amplitude
    detuning = 0

    match direction:
        case "x":
            phase = 0
        case "y":
            phase = np.pi / 2
        case _:
            phase = direction

    sequence.enable_eom_mode(support_list, amp_on=amplitude, detuning_on=detuning)
    sequence.add_eom_pulse(support_list, duration=int(duration), phase=phase)
    sequence.disable_eom_mode(support_list)


def free_evolution(
    sequence: Sequence,
    support: Support,
    duration: Scalar,
    *_args: Any,
    **_kwargs: Any,
) -> None:
    support_list = "LOCAL" if support.target else "GLOBAL"
    max_amp = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    duration *= 1000 * 2 * np.pi / max_amp
    sequence.delay(int(duration), support_list)  # type: ignore


def apply_local_shifts(sequence: Sequence, support: Support, **_: Any) -> None:
    max_abs_detuning = (
        sequence.device.channels["rydberg_global"].max_abs_detuning or DEFAULT_DETUNING
    )
    time_scale = 1000 * 2 * np.pi / max_abs_detuning
    _local_pulse_core(
        sequence, support, duration=1.0, time_scale=time_scale, detuning=1.0, concurrent=False
    )


def local_pulse(
    sequence: Sequence,
    support: Support,
    duration: Scalar | Literal["fill"],
    detuning: Scalar,
    concurrent: bool = False,
    **_: Any,
) -> None:
    max_amp = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    time_scale = 1000 * 2 * np.pi / max_amp
    _local_pulse_core(sequence, support, duration, time_scale, detuning, concurrent)


def _local_pulse_core(
    sequence: Sequence,
    support: Support,
    duration: Scalar | Literal["fill"],
    time_scale: float,
    detuning: Scalar,
    concurrent: bool = False,
    **_kwargs: Any,
) -> None:
    support_list = "LOCAL" if support.target else "GLOBAL"
    max_abs_detuning = (
        sequence.device.channels["rydberg_global"].max_abs_detuning or DEFAULT_DETUNING
    )

    if duration == "fill":
        if not concurrent:
            raise SyntaxError(
                "The option `fill` can only be used on the `concurrent` mode"
            )

        duration = sequence.get_duration(support_list) - sequence.get_duration("dmm_0")
    else:
        duration *= time_scale  # type: ignore

    detuning *= -1 * max_abs_detuning

    sequence.add_dmm_detuning(
        ConstantWaveform(duration, detuning),
        "dmm_0",
        "no-delay" if concurrent else "min-delay",
    )


def h_pulse(
    sequence: Sequence,
    support: Support,
    duration: Scalar = 1000.0,
    **_: Any,
) -> None:
    support_list = "LOCAL" if support.target else "GLOBAL"
    amplitude = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    duration *= 1000 * 2 * np.pi / amplitude

    sequence.enable_eom_mode(support_list, amp_on=amplitude, correct_phase_drift=True)
    sequence.add_eom_pulse(
        support_list,
        duration=int(duration),
        phase=np.pi / 2,
        post_phase_shift=np.pi
    )
    sequence.disable_eom_mode(support_list)
