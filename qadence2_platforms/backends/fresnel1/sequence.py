from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from pulser.register.register_layout import RegisterLayout
from pulser.sequence.sequence import Sequence
from qadence2_ir.types import Model

from qadence2_platforms.backends._base_analog.sequence import from_instructions

from . import functions as add_pulse
from .device_settings import Fresnel1Settings
from .functions import PULSE_FN_MAP


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
