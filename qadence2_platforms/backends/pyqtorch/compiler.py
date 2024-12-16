from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger

import pyqtorch as pyq
import torch
from pyqtorch.quantum_operation import QuantumOperation
from qadence2_ir.types import Alloc, Load, Model, QuInstruct

from qadence2_platforms.backends.pyqtorch.embedding import Embedding
from qadence2_platforms.backends.pyqtorch.interface import Interface
from qadence2_platforms.backends.pyqtorch.register import RegisterInterface

logger = getLogger(__name__)


class Compiler:
    """
    A dataclass to compile an IR model data into PyQTorch objects (to be run in a
    PyQTorch-based backend.
    """

    instruction_mapping = {
        "not": pyq.CNOT,
        "add": pyq.Add,
        "mul": pyq.Scale,
        "noncommute": pyq.Sequence,
    }

    @classmethod
    def _get_target(cls, target: tuple[int, ...] | tuple, num_qubits: int) -> tuple[int, ...]:
        return tuple(range(num_qubits)) if len(target) == 0 else target

    def compile(
        self,
        model: Model,
    ) -> pyq.QuantumCircuit:
        """
        Compiling IR model data to PyQTorch object function. It transforms model
        `QuInstruct`s into PyQTorch operators, resolving the SSA-form arguments
        into concrete values or valid PyQTorch parameters.

        Args:
            model (Model): IR model to compile

        Returns:
            A PyQTorch quantum circuit object with the model `QuInstruct`s compiled into
            PyQTorch operators
        """

        pyq_operations = []

        for instr in model.instructions:

            if isinstance(instr, QuInstruct):
                native_op: QuantumOperation = getattr(
                    pyq, instr.name.upper(), self.instruction_mapping.get(instr.name)
                )
                control = instr.support.control
                target = self._get_target(instr.support.target, model.register.num_qubits)
                native_support = (*control, *target)

                if len(instr.args) > 0:
                    assert len(instr.args) == 1, "More than one arg not supported"
                    (maybe_load,) = instr.args
                    assert isinstance(maybe_load, Load), "only support load"
                    pyq_operations.append(
                        native_op(native_support, maybe_load.variable).to(dtype=torch.complex128)
                    )

                else:
                    pyq_operations.append(native_op(*native_support).to(dtype=torch.complex128))

        return pyq.QuantumCircuit(model.register.num_qubits, pyq_operations).to(
            dtype=torch.complex128
        )


def get_trainable_params(inputs: dict[str, Alloc]) -> dict[str, torch.Tensor]:
    return {
        param: torch.rand(value.size, requires_grad=True)
        for param, value in inputs.items()
        if value.is_trainable
    }


def compile_to_backend(model: Model) -> Interface:
    """
    Compiles the model data (IR information from expressions) into PyQTorch-compatible data and
    defines an Interface instance to be available to the user to invoke useful methods, such as
    `run`, `sample`, `expectation`, `set_parameters`.

    Args:
        model (Model): the IR model data to be compiled to PyQTorch-based backend

    Returns:
        The `Interface` instance based on PyQTorch backend
    """

    register_interface = RegisterInterface(
        model.register.num_qubits, model.register.options.get("init_state")
    )
    embedding = Embedding(model)
    native_circ = Compiler().compile(model)
    vparams = get_trainable_params(model.inputs)
    return Interface(register_interface, embedding, native_circ, vparams=vparams)
