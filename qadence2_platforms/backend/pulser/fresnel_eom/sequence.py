from __future__ import annotations

from typing import Callable

import numpy as np
from pulser.devices._device_datacls import BaseDevice
from pulser.register.base_register import BaseRegister
from pulser.sequence import Sequence as PulserSequence

from qadence2_platforms import Model
from qadence2_platforms.backend.sequence import SequenceApi
from qadence2_platforms.qadence_ir import QuInstruct

from ..backend import SequenceType
from .instructions import h_fn, not_fn, qubit_dyn_fn, rx_fn


class Sequence(SequenceApi[SequenceType, BaseRegister, BaseDevice]):
    instruction_map: dict[str, Callable] = {
        "not": not_fn,
        "h": h_fn,
        "rx": rx_fn,
        "qubit_dyn": qubit_dyn_fn,
    }

    def __init__(self, model: Model, device: BaseDevice, register: BaseRegister):
        self.model: Model = model
        self.device: BaseDevice = device
        self.register: BaseRegister = register

    def _define_sequence(self) -> PulserSequence:
        seq = PulserSequence(self.register, self.device)
        seq.declare_channel("GLOBAL", "rydberg_global")

        if self.model.directives.get("local_targets"):
            if self.model.directives.get("local_shifts"):
                targets = zip(
                    self.model.directives["local_targets"],
                    self.model.directives["local_shifts"],
                )
                detuning_map = self.register.define_detuning_map(
                    {elem[0]: elem[1] / (2 * np.pi) for elem in targets}
                )
            else:
                targets = zip(
                    self.model.directives["local_targets"],
                    [1] * len(self.model.directives["local_targets"]),
                )
                detuning_map = self.register.define_detuning_map(dict(targets))

            seq.config_detuning_map(detuning_map, "dmm_0")
        return seq

    def build_sequence(self) -> SequenceType:
        seq = self._define_sequence()
        pulses_list = []

        for instr in self.model.instructions:
            if isinstance(instr, QuInstruct):
                native_op = self.instruction_map[instr.name](
                    seq=seq, support=instr.support
                )
                pulses_list.append(native_op)
        return pulses_list
