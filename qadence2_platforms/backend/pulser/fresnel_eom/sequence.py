from __future__ import annotations

from functools import lru_cache, partial
from typing import Any, Callable, Optional, cast

import numpy as np
from pulser.devices._device_datacls import BaseDevice
from pulser.register.base_register import BaseRegister
from pulser.sequence import Sequence as PulserSequence

from qadence2_platforms import Model
from qadence2_platforms.backend.sequence import SequenceApi
from qadence2_platforms.qadence_ir import QuInstruct

from ..backend import InstructLazyResult
from ..embedding import EmbeddingModule
from .instructions import h_fn, not_fn, qubit_dyn_fn, rx_fn


class BackendLazySequence:
    """
    Lazy sequence class to hold a tuple of instructions lazy results class to be
    evaluated at runtime, once the parameters are provided by the user or through
    time-dependent steps.
    """

    def __init__(self, *instructions: Any):
        self.lazy_instr: tuple[InstructLazyResult, ...] = instructions

    @staticmethod
    @lru_cache
    def get_fn_args(fn: partial) -> Any:
        match fn.func.__name__:
            case "rotation":
                return ["angle", "direction"]
            case "pulse":
                return ["duration", "amplitude", "detuning", "phase"]
            case "free_evolution":
                return ["duration"]
            case _:
                return []

    def evaluate(
        self, embedding: EmbeddingModule, values: Optional[dict] = None
    ) -> PulserSequence:
        seq: Optional[PulserSequence] = None
        assigned_values: dict = embedding(values)
        for fn, params in self.lazy_instr:
            resolved_params: tuple[Any, ...] = ()
            params = cast(tuple, params)
            fn = cast(partial, fn)
            for param in params:
                resolved_params += (assigned_values[param.variable],)
            seq = fn(**dict(zip(self.get_fn_args(fn), resolved_params)))

        if seq:
            return seq
        raise ValueError("pulser sequence must not be None.")


class Sequence(SequenceApi[BackendLazySequence, BaseRegister, BaseDevice]):
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

    def build_sequence(self) -> BackendLazySequence:
        seq = self._define_sequence()
        pulses_list = []

        for instr in self.model.instructions:
            if isinstance(instr, QuInstruct):
                native_op = self.instruction_map[instr.name](
                    seq=seq, support=instr.support, params=instr.args
                )
                pulses_list.append(native_op)
        return BackendLazySequence(*pulses_list)
