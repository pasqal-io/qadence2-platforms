from __future__ import annotations

from collections import Counter
from types import ModuleType
from typing import Any, Optional

import pyqtorch as pyq
import torch

from qadence2.platforms.backend.interface import RuntimeInterfaceApi

from .embedding import EmbeddingModule
from .register import RegisterInterface

# TODO: resolve the `observable` attribute


class RuntimeInterface(
    torch.nn.Module,
    RuntimeInterfaceApi[
        RegisterInterface,
        EmbeddingModule,
        pyq.QuantumCircuit,
        ModuleType,
        torch.Tensor,
        list[Counter],
        torch.Tensor,
        torch.Tensor,
    ],
):
    def __init__(
        self,
        register: RegisterInterface,
        embedding: EmbeddingModule,
        native_seq: pyq.QuantumCircuit,
        native_backend: ModuleType,
        observable: Any = None,
    ):
        super().__init__()
        self.register: RegisterInterface = register
        self.embedding: EmbeddingModule = embedding
        self.sequence: pyq.QuantumCircuit = native_seq
        self.engine: ModuleType = native_backend
        self.observable: Any = observable

    def forward(
        self, state: torch.Tensor, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if state is None:
            state = self.init_state
        return self.engine.run(self.sequence, state, self.embedding(inputs))

    def run(
        self,
        state: Optional[torch.Tensor] = None,
        inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        return self.forward(state, inputs or dict())

    def sample(
        self, state: torch.Tensor, inputs: dict[str, torch.Tensor], n_shots: int = 1000
    ) -> list[Counter]:
        return self.engine.sample(self.sequence, state, self.embedding(inputs), n_shots)

    def expectation(
        self, state: torch.Tensor, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.engine.expectation(
            self.sequence, state, self.embedding(inputs), self.observable
        )
