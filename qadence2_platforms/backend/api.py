from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from importlib import import_module
from logging import getLogger
from typing import Any, Generic

import torch

from qadence2_platforms.qadence_ir import Model
from qadence2_platforms.types import (
    ArgsType,
    BackendInstructResultType,
    DeviceType,
    DirectivesType,
    RegisterType,
    SequenceObjectType,
    SupportType,
)

logger = getLogger(__name__)


class Api(torch.nn.Module):
    """A class holding register,embedding, circuit, native backend and optional observable."""

    def __init__(
        self,
        register: Any,
        embedding: Any,
        circuit: Any,
        native_backend: Any,
        observable: Any = None,
    ) -> None:
        super().__init__()
        self.register = register
        self.init_state = (
            circuit.from_bitstring(register.init_state)
            if register.init_state is not None
            else circuit.init_state()
        )
        self.embedding = embedding
        self.circuit = circuit
        self.observable = observable
        self.native_backend = native_backend

    @property
    def n_qubits(self) -> int:
        return self.register.n_qubits  # type: ignore[no-any-return]

    def forward(
        self, state: torch.Tensor, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if state is None:
            state = self.init_state
        return self.native_backend.run(self.circuit, state, self.embedding(inputs))

    def run(
        self, state: torch.Tensor = None, inputs: dict[str, torch.Tensor] = dict()
    ) -> torch.Tensor:
        return self.forward(state, inputs)

    def sample(
        self, state: torch.Tensor, inputs: dict[str, torch.Tensor], n_shots: int = 1000
    ) -> list[Counter]:
        return self.native_backend.sample(self.circuit, state, self.embedding(inputs), n_shots)  # type: ignore[no-any-return]

    def expectation(
        self, state: torch.Tensor, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.native_backend.expectation(
            self.circuit, state, self.embedding(inputs), self.observable
        )


def compile(model: Model, backend_name: str) -> Api:  # type: ignore[return]
    try:
        interface = import_module(f"qadence2_platforms.backend.{backend_name}")
        native_backend = import_module(backend_name)
        register_interface = interface.RegisterInterface(model)
        embedding = interface.Embedding(model)
        native_circ = interface.Compiler().compile(model)
        return Api(register_interface, embedding, native_circ, native_backend)
    except Exception as e:
        logger.error(f"Unable to import backend {backend_name} due to {e}.")

class BackendSequenceAPI(
    ABC, Generic[RegisterType, DeviceType, DirectivesType, SequenceObjectType]
):
    @abstractmethod
    def get_sequence(self) -> SequenceObjectType: ...
