from __future__ import annotations

from logging import getLogger
from typing import Any, Callable, Counter, Literal, Optional

import pyqtorch
import torch
from qadence2_ir.types import Model

from qadence2_platforms.abstracts import (
    AbstractInterface,
)

from .compiler import Compiler
from .embedding import Embedding
from .register import RegisterInterface

logger = getLogger(__name__)


class Interface(
    AbstractInterface[
        torch.Tensor,
        pyqtorch.QuantumCircuit,
        torch.Tensor,
        torch.Tensor,
        list[Counter],
        torch.Tensor,
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

    def set_parameters(self, params: dict[str, float]) -> None:
        pass

    def _run(
        self,
        run_type: Literal["run", "sample", "expectation"],
        values: Optional[dict[str, torch.Tensor]] = None,
        callback: Optional[Callable] = None,
        state: torch.Tensor | None = None,
        shots: int | None = None,
        observable: Any | None = None,
        **_: Any,
    ) -> Any:
        """
        Method to execute run type-specific option. Option can be `run`, `sample`
        or `expectation`. Each option can have different arguments, such as:
        `sample` uses `shots`, while `expectation` uses `observable`.

        It should not be called directly. Use it on `run`, `sample` or `expectation`
        methods.

        :param run_type: str option as `run`, `sample` or `expectation`
        :param values: dictionary of user-input parameters
        :param callback: callback function to be used internally, if applicable
        :param state: a tensor containing the desired state to perform the execution from
        :param shots: number of shots, if applicable (`sample` only)
        :param observable: a list of observables, if applicable (`expectation` only)
        :return: a tensor or list of values (`sample` only) of the calculated state
        """
        inputs = values or dict()
        state = state if state is not None else self.init_state

        match run_type:
            case "run":
                return pyqtorch.run(
                    circuit=self.circuit,
                    state=state,
                    values=inputs,
                    embedding=self.embedding,
                )
            case "sample":
                return pyqtorch.sample(
                    circuit=self.circuit,
                    state=state,
                    values=inputs,
                    n_shots=shots,
                    embedding=self.embedding,
                )
            case "expectation":
                if observable is not None:
                    return pyqtorch.expectation(
                        circuit=self.circuit,
                        state=state,
                        values=inputs,
                        observable=self.observable,
                        embedding=self.embedding,
                    )
                raise ValueError("Observable must not be None for expectation run.")
            case _:
                raise NotImplementedError(f"Run type '{run_type}' not implemented.")

    def run(
        self,
        *,
        values: Optional[dict[str, torch.Tensor]] = None,
        callback: Optional[Callable] = None,
        state: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._run("run", values=values, callback=callback, state=state, **kwargs)

    def sample(
        self,
        *,
        values: Optional[dict[str, torch.Tensor]] = None,
        shots: Optional[int] = None,
        callback: Optional[Callable] = None,
        state: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> list[Counter]:
        return self._run(
            "sample",
            values=values,
            callback=callback,
            shots=shots,
            state=state,
            **kwargs,
        )

    def expectation(
        self,
        *,
        values: Optional[dict[str, torch.Tensor]] = None,
        callback: Optional[Callable] = None,
        state: torch.Tensor | None = None,
        observable: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._run(
            "expectation",
            values=values,
            callback=callback,
            state=state,
            observable=observable,
            **kwargs,
        )


def compile_to_backend(model: Model) -> Interface:
    register_interface = RegisterInterface(
        model.register.num_qubits, model.register.options.get("init_state")
    )
    embedding = Embedding(model)
    native_circ = Compiler().compile(model)
    return Interface(register_interface, embedding, native_circ)
