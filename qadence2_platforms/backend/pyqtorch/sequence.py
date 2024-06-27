from __future__ import annotations

from typing import Callable, Any

import pyqtorch as pyq

from qadence2_platforms import Model
from qadence2_platforms.backend.sequence import SequenceApi
from qadence2_platforms.qadence_ir import QuInstruct, Load


class Sequence(SequenceApi[pyq.QuantumCircuit]):
    instruction_map: dict[str, Callable] = {
        "not": pyq.CNOT,
        "add": pyq.Add,
        "mul": pyq.Scale,
        "noncommute": pyq.Sequence,
    }

    def __init__(self, model: Model, **_: Any):
        self.model: Model = model

    def build_sequence(self) -> pyq.QuantumCircuit:
        pyq_operations = []
        for instr in self.model.instructions:
            if isinstance(instr, QuInstruct):
                native_op = getattr(pyq, instr.name.upper(), None)
                if native_op is None:
                    native_op = self.instruction_map[instr.name]
                control = instr.support.control
                target = instr.support.target
                native_support = (*control, *target)
                if len(instr.args) > 0:
                    assert len(instr.args) == 1, "More than one arg not supported"
                    (maybe_load,) = instr.args
                    assert isinstance(maybe_load, Load), "only support load"
                    pyq_operations.append(
                        native_op(native_support, maybe_load.variable)
                    )
                else:
                    pyq_operations.append(native_op(*native_support))
        return pyq.QuantumCircuit(self.model.register.num_qubits, pyq_operations)
