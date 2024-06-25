from __future__ import annotations

from typing import Any
from functools import partial

import numpy as np
from pulser.sequence import Sequence
from pulser.devices._device_datacls import BaseDevice
from pulser.register.base_register import BaseRegister

from qadence2_platforms.qadence_ir import Load, Support
from qadence2_platforms.backend.api import QuInstructAPI, BackendSequenceAPI
from qadence2_platforms.backend.utils import BackendInstructResult
from .functions import rotation, h_pulse, free_evolution


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


class BackendInstruct(QuInstructAPI[Sequence]):
    @classmethod
    def not_fn(
        cls,
        seq: Sequence,
        support: Support,
        *args: Load,
        **options: Any,
    ) -> BackendInstructResult:
        return BackendInstructResult(
            fn=partial(
                rotation, sequence=seq, support=support, angle=np.pi, direction="x"
            ),
            *args
        )

    @classmethod
    def z_fn(
        cls,
        seq: Sequence,
        support: Support,
        *args: Load,
        **options: Any
    ) -> BackendInstructResult:
        ...

    @classmethod
    def h_fn(
        cls,
        seq: Sequence,
        support: Support,
        *args: Load,
        **options: Any
    ) -> BackendInstructResult:
        return BackendInstructResult(
            fn=partial(h_pulse, sequence=seq, support=support),
            *args
        )

    @classmethod
    def rx_fn(
        cls,
        seq: Sequence,
        support: Support,
        *args: Load,
        **options: Any
    ) -> BackendInstructResult:
        return BackendInstructResult(
            fn=partial(rotation, sequence=seq, support=support, direction="x"),
            *args
        )

    @classmethod
    def qubit_dyn_fn(
        cls,
        seq: Sequence,
        support: Support,
        *args: Load,
        **options: Any
    ) -> BackendInstructResult:
        # for the sake of testing purposes, qubit dynamics will be only
        # a simple free evolution pulse
        return BackendInstructResult(
            fn=partial(free_evolution, sequence=seq, support=support),
            *args
        )
