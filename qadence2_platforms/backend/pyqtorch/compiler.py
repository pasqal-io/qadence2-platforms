from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger

import pyqtorch as pyq

from qadence2_platforms.qadence_ir import Load, Model, QuInstruct

logger = getLogger(__name__)


@dataclass(frozen=True)
class Compiler:
    instruction_mapping = {
        "not": pyq.CNOT,
        "add": pyq.Add,
        "mul": pyq.Scale,
        "noncommute": pyq.Sequence,
    }

    def compile(
        self,
        model: Model,
    ) -> pyq.QuantumCircuit:
        pyq_operations = []
        for instr in model.instructions:
            if isinstance(instr, QuInstruct):
                native_op = None
                try:
                    native_op = getattr(pyq, instr.name.upper())
                except Exception as e:
                    native_op = self.instruction_mapping[instr.name]
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
        return pyq.QuantumCircuit(model.register.num_qubits, pyq_operations)
