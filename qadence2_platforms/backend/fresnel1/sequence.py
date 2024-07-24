from __future__ import annotations


from dataclasses import replace
from typing import Any, Iterable
import numpy as np

from pulser.channels import DMM
from pulser.devices._devices import AnalogDevice
from pulser.sequence.sequence import Sequence
from pulser.register.register_layout import RegisterLayout

from qadence2_ir.types import Model, Assign, QuInstruct, Call, Load

from . import functions as add_pulse


class NamedPulse:
    def __init__(self, name: str, args: tuple[str | float, ...]) -> None:
        self.name = name
        self.args = args


# Fresnel1 = AnalogDevice
Fresnel1 = replace(
    AnalogDevice.to_virtual(),
    dmm_objects=(
        DMM(
            # from Pulser tutorials/dmm.html#DMM-Channel-and-Device
            clock_period=4,
            min_duration=16,
            max_duration=2**26,
            mod_bandwidth=8,
            bottom_detuning=-2 * np.pi * 20,  # detuning between 0 and -20 MHz
            total_bottom_detuning=-2 * np.pi * 2000,  # total detuning
        ),
    ),
)


def to_sequence(model: Model, register: RegisterLayout) -> Sequence:
    # TODO: fix the map implementation
    return NotImplemented

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

    for pulse in pulses:
        getattr(add_pulse, pulse.name)(seq, **args)  # type: ignore

    return seq
