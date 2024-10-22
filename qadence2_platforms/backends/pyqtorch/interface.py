from __future__ import annotations

from logging import getLogger
from typing import Any, Callable, Counter, Literal, Optional, cast

import pyqtorch as pyq
import torch

from qadence2_platforms.abstracts import (
    AbstractInterface,
    RunEnum,
)

from .embedding import Embedding
from .functions import load_observables
from .register import RegisterInterface
from .utils import InputType

logger = getLogger(__name__)


class Interface(
    AbstractInterface[
        torch.Tensor,
        pyq.QuantumCircuit,
        torch.Tensor,
        torch.Tensor,
        list[Counter],
        torch.Tensor,
    ],
):
    """A class holding register, embedding, circuit, native backends and optional observable."""

    def __init__(
        self,
        register: RegisterInterface,
        embedding: Embedding,
        circuit: pyq.QuantumCircuit,
        observable: list[InputType] | InputType | None = None,
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
    def sequence(self) -> pyq.QuantumCircuit:
        return self.circuit

    def add_noise(self, model: Literal["SPAM"]) -> None:
        pass

    def set_parameters(self, params: dict[str, float]) -> None:
        pass

    def _run(
        self,
        run_type: RunEnum,
        values: dict[str, torch.Tensor] | None = None,
        callback: Optional[Callable] = None,
        state: torch.Tensor | None = None,
        shots: int | None = None,
        observable: list[InputType] | InputType | None = None,
        **_: Any,
    ) -> Any:
        """
        Method to execute run type-specific option.

        Option can be `run`, `sample`
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
            case RunEnum.RUN:
                return pyq.run(
                    circuit=self.circuit,
                    state=state,
                    values=inputs,
                    embedding=self.embedding,
                )
            case RunEnum.SAMPLE:
                return pyq.sample(
                    circuit=self.circuit,
                    state=state,
                    values=inputs,
                    n_shots=shots,
                    embedding=self.embedding,
                )
            case RunEnum.EXPECTATION:
                if observable is not None or self.observable is not None:
                    return pyq.expectation(
                        circuit=self.circuit,
                        state=state,
                        values=inputs,
                        observable=load_observables(observable or self.observable),  # type: ignore [arg-type]
                        embedding=self.embedding,
                    )
                raise ValueError("Observable must not be None for expectation run.")
            case _:
                raise NotImplementedError(f"Run type '{run_type}' not implemented.")

    def run(
        self,
        *,
        values: dict[str, torch.Tensor] | None = None,
        callback: Optional[Callable] = None,
        state: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._run(RunEnum.RUN, values=values, callback=callback, state=state, **kwargs)

    def sample(
        self,
        *,
        values: dict[str, torch.Tensor] | None = None,
        shots: int | None = None,
        callback: Optional[Callable] = None,
        state: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> list[Counter]:
        return cast(
            list,
            self._run(
                RunEnum.SAMPLE,
                values=values,
                callback=callback,
                shots=shots,
                state=state,
                **kwargs,
            ),
        )

    def expectation(
        self,
        *,
        values: dict[str, torch.Tensor] | None = None,
        callback: Optional[Callable] = None,
        state: torch.Tensor | None = None,
        observable: list[InputType] | InputType | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._run(
            RunEnum.EXPECTATION,
            values=values,
            callback=callback,
            state=state,
            observable=observable,
            **kwargs,
        )
