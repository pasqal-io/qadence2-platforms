from __future__ import annotations

from collections import Counter
from types import ModuleType
from typing import Any

import numpy as np
from pulser.register.base_register import BaseRegister
from pulser_simulation import QutipEmulator

from qadence2_platforms.backend.interface import RuntimeInterfaceApi

from .embedding import EmbeddingModule
from .backend import SequenceType
from ...types import ExpectationResultType


class RuntimeInterface(
    RuntimeInterfaceApi[
        BaseRegister,
        EmbeddingModule,
        SequenceType,
        ModuleType,
        np.ndarray,
        Counter,
        np.ndarray,
        np.ndarray
    ]
):
    def __init__(
        self,
        register: BaseRegister,
        embedding: EmbeddingModule,
        native_seq: SequenceType,
        native_backend: ModuleType,
        **_: Any
    ):
        self.register: BaseRegister = register
        self.embedding: EmbeddingModule = embedding
        self.sequence: SequenceType = native_seq
        self.engine: ModuleType = native_backend

    def __call__(self, **_: Any) -> np.ndarray:
        return self.forward()

    def forward(self, **_: Any) -> np.ndarray:
        raise NotImplementedError()

    def run(self, num_shots: int = 1000, on: str = "emulator") -> Counter:
        match on:
            case "emulator":
                simulation = QutipEmulator.from_sequence(
                    self.sequence, with_modulation=True
                )
                result = simulation.run()
                return result.sample_final_state(N_samples=num_shots)
            case _:
                raise NotImplementedError("only emulator mode available on Pulser engine.")

    def sample(self, num_shots: int) -> Counter:
        return self.run(num_shots)

    def expectation(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError()
