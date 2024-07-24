from __future__ import annotations

from logging import getLogger
from typing import Any, Callable, Counter, Literal

import torch
import pyqtorch

from qadence2_ir.types import Model
from qadence2_platforms.abstracts import AbstractInterface

from .register import RegisterInterface
from .embedding import Embedding
from .compiler import Compiler

logger = getLogger(__name__)


class Interface(
    AbstractInterface[
        pyqtorch.QuantumCircuit, torch.Tensor, torch.Tensor | list[Counter]
    ],
):
    """A class holding register,embedding, circuit, native backend and optional observable."""

    def __init__(
        self,
        register: RegisterInterface,
        embedding: Embedding,
        circuit: pyqtorch.QuantumCircuit,
        observable: Any = None,
    ) -> None:
        super().__init__()
        self.register = register
        self.init_state: torch.Tensor = (
            circuit.from_bitstring(register.init_state)
            if register.init_state is not None
            else circuit.init_state()
        )
        self.embedding = embedding
        self.circuit = circuit
        self.observable = observable

    @property
    def info(self) -> dict[str, Any]:
        return {"num_qubits": self.register.n_qubits}

    @property
    def sequence(self) -> pyqtorch.QuantumCircuit:
        return self.circuit

    def add_noise(self, model: Literal["SPAM"]) -> None:
        pass

    def run(
        self,
        *,
        parameters: dict[str, torch.Tensor] | None = None,
        shots: int | None = None,
        callback: Callable | None = None,
        state: torch.Tensor | None = None,
        observable: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | list[Counter]:
        inputs = parameters or dict()
        state = state or self.init_state

        # Expectation
        if observable:
            return pyqtorch.expectation(
                circuit=self.circuit,
                state=state,
                values=inputs,
                observable=self.observable,
                embedding=self.embedding,
            )

        # Simulation
        if not shots:
            return pyqtorch.run(
                circuit=self.circuit,
                state=state,
                values=inputs,
                embedding=self.embedding,
            )

        # Sample
        return pyqtorch.sample(
            circuit=self.circuit,
            state=state,
            values=inputs,
            n_shots=shots,
            embedding=self.embedding,
        )


def build(model: Model) -> Interface:
    register_interface = RegisterInterface(
        model.register.num_qubits, model.register.options.get("init_state")
    )
    embedding = Embedding(model)
    native_circ = Compiler().compile(model)
    return Interface(register_interface, embedding, native_circ)
