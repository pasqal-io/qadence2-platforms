from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger

import pyqtorch as pyq
import torch
from qadence2_ir.types import Load, Model, QuInstruct, Alloc

from qadence2_platforms.backends.pyqtorch.embedding import Embedding
from qadence2_platforms.backends.pyqtorch.interface import Interface
from qadence2_platforms.backends.pyqtorch.register import RegisterInterface

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
                except Exception as _:
                    native_op = self.instruction_mapping[instr.name]
                control = instr.support.control
                target = instr.support.target
                native_support = (*control, *target)
                if len(instr.args) > 0:
                    assert len(instr.args) == 1, "More than one arg not supported"
                    (maybe_load,) = instr.args
                    assert isinstance(maybe_load, Load), "only support load"
                    pyq_operations.append(native_op(native_support, maybe_load.variable))
                else:
                    pyq_operations.append(native_op(*native_support))
        return pyq.QuantumCircuit(model.register.num_qubits, pyq_operations)


def _get_trainable_params(inputs: dict[str, Alloc]) -> dict[str, torch.Tensor]:
    return {
        param: torch.rand(value.size, requires_grad=True)
        for param, value in inputs.items()
            if value.is_trainable
    }

def compile_to_backend(model: Model) -> Interface:
    register_interface = RegisterInterface(
        model.register.num_qubits, model.register.options.get("init_state")
    )
    embedding = Embedding(model)
    native_circ = Compiler().compile(model)
    vparams = _get_trainable_params(model.inputs)
    return Interface(register_interface, embedding, native_circ, vparams=vparams)
