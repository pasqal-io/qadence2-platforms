from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
from pulser.devices._device_datacls import BaseDevice
from pulser.register.base_register import BaseRegister
from pulser.sequence import Sequence

from qadence2_platforms.backend.api import BackendSequenceAPI
from ..backend import BackendInstructResult
from qadence2_platforms.qadence_ir import Support

from .functions import free_evolution, h_pulse, rotation


class BackendSequence(BackendSequenceAPI[BaseRegister, BaseDevice, Sequence, dict]):
    def __init__(self, register: BaseRegister, device: BaseDevice, directives: dict):
        self.register: BaseRegister = register
        self.device: BaseDevice = device
        self.directives: dict = directives

    def get_sequence(self) -> Sequence:
        seq: Sequence = Sequence(self.register, self.device)
        seq.declare_channel("global", "rydberg_global")

        if self.directives.get("local_targets"):
            if self.directives.get("local_shifts"):
                targets = zip(
                    self.directives["local_targets"],
                    self.directives["local_shifts"],
                )
                detuning_map = self.register.define_detuning_map(
                    {elem[0]: elem[1] / (2 * np.pi) for elem in targets}
                )
            else:
                targets = zip(
                    self.directives["local_targets"],
                    [1] * len(self.directives["local_targets"]),
                )
                detuning_map = self.register.define_detuning_map(dict(targets))

            seq.config_detuning_map(detuning_map, "dmm_0")
        return seq


def not_fn(
    seq: Sequence,
    support: Support,
    *args: Any,
    **_: Any,
) -> BackendInstructResult:
    return BackendInstructResult(
        fn=partial(
            rotation, sequence=seq, support=support, angle=np.pi, direction="x"
        ),
        *args,
    )


def h_fn(seq: Sequence, support: Support, *args: Any, **_: Any) -> BackendInstructResult:
    return BackendInstructResult(
        fn=partial(h_pulse, sequence=seq, support=support), *args
    )


def rx_fn(seq: Sequence, support: Support, *args: Any, **_: Any) -> BackendInstructResult:
    return BackendInstructResult(
        fn=partial(rotation, sequence=seq, support=support, direction="x"), *args
    )


def qubit_dyn_fn(
    seq: Sequence, support: Support, *args: Any, **_: Any
) -> BackendInstructResult:
    # for the sake of testing purposes, qubit dynamics will be only
    # a simple free evolution pulse
    return BackendInstructResult(
        fn=partial(free_evolution, sequence=seq, support=support), *args
    )
