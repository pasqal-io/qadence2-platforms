from __future__ import annotations

from collections import Counter
from importlib import import_module
from logging import getLogger

import pyqtorch as pyq
import torch

from qadence2_platforms.backend.pyqtorch.embed import Embedding
from qadence2_platforms.qadence_ir import Model

logger = getLogger(__name__)


class Api(torch.nn.Module):
    """A class holding the final embedding instance and the pyq.QuantumCircuit."""

    def __init__(
        self,
        embedding: Embedding,
        circuit: pyq.QuantumCircuit,
        observable: pyq.Observable = None,
        backend: str = "pyqtorch",
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.circuit = circuit
        self.observable = observable
        self.backend = backend

    def forward(
        self, state: torch.Tensor, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.run(state, inputs)

    def run(self, state: torch.Tensor, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return pyq.run(self.circuit, state, self.embedding(inputs))

    def sample(
        self, state: torch.Tensor, inputs: dict[str, torch.Tensor], n_shots: int = 1000
    ) -> list[Counter]:
        return pyq.sample(self.circuit, state, self.embedding(inputs), n_shots)  # type: ignore[no-any-return]

    def expectation(
        self, state: torch.Tensor, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return pyq.expectation(
            self.circuit, state, self.embedding(inputs), self.observable
        )


def compile(model: Model, backend: str) -> Api:
    embed = import_module(f"qadence2_platforms.backend.{backend}.embed")
    compiler = import_module(f"qadence2_platforms.backend.{backend}.compile")
    embedding = embed.Embedding(model)
    native_circ = compiler.compile(model)
    return Api(embedding, native_circ, backend)
