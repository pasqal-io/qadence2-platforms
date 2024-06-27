from __future__ import annotations

from collections import Counter
from types import ModuleType
from typing import Any, Optional

import numpy as np
from pulser.register.base_register import BaseRegister
from pulser_simulation import QutipEmulator

from qadence2_platforms.backend.interface import RuntimeInterfaceApi

from .embedding import EmbeddingModule
from .fresnel_eom.sequence import BackendPartialSequence


class RuntimeInterface(
    RuntimeInterfaceApi[
        BaseRegister,
        EmbeddingModule,
        BackendPartialSequence,
        ModuleType,
        np.ndarray,
        Counter,
        np.ndarray,
        np.ndarray,
    ]
):
    def __init__(
        self,
        register: BaseRegister,
        embedding: EmbeddingModule,
        native_seq: BackendPartialSequence,
        native_backend: ModuleType,
        **_: Any,
    ):
        self.register: BaseRegister = register
        self.embedding: EmbeddingModule = embedding
        self.sequence: BackendPartialSequence = native_seq
        self.engine: ModuleType = native_backend

    def __call__(self, **_: Any) -> np.ndarray:
        return self.forward()

    def forward(self, **_: Any) -> np.ndarray:
        raise NotImplementedError()

    def run(
        self, num_shots: int = 1000, on: str = "emulator", values: Optional[dict] = None
    ) -> Counter:
        values = values or dict()
        match on:
            case "emulator":
                simulation = QutipEmulator.from_sequence(
                    sequence=self.sequence.evaluate(self.embedding, values),
                    with_modulation=True,
                )
                result = simulation.run()
                return result.sample_final_state(N_samples=num_shots)
            case _:
                raise NotImplementedError(
                    "only emulator mode available on Pulser engine."
                )

    def sample(
        self, num_shots: int, on: str = "emulator", values: Optional[dict] = None
    ) -> Counter:
        return self.run(num_shots, on, values)

    def expectation(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError()
