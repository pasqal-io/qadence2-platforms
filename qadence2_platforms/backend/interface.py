from __future__ import annotations

from typing import Any, Generic

from qadence2_platforms.backend.bytecode import Bytecode
from qadence2_platforms.backend.utils import get_backend_module
from qadence2_platforms.types import (
    ExpectationResultType,
    InterfaceCallResultType,
    InterfaceInstructType,
    RunResultType,
    SampleResultType,
)


class Interface(
    Generic[
        InterfaceInstructType,
        RunResultType,
        SampleResultType,
        ExpectationResultType,
        InterfaceCallResultType,
    ]
):
    def __init__(self, bytecode: Bytecode):
        self.bytecode: Bytecode = bytecode
        self.instructions: tuple[InterfaceInstructType, ...] = (
            self._resolve_parameters()
        )

    def _resolve_parameters(self) -> Any:
        resolve_fn = getattr(
            get_backend_module(self.bytecode.backend), "resolve_parameters"
        )
        return resolve_fn(
            instructions=self.bytecode.instructions, variables=self.bytecode.variables
        )

    def __call__(self, *args: Any, **kwargs: Any) -> InterfaceCallResultType:
        raise NotImplementedError()

    def forward(self, *args: Any, **kwargs: Any) -> InterfaceCallResultType:
        return self.__call__(*args, **kwargs)

    def run(self, **kwargs: Any) -> RunResultType:
        raise NotImplementedError()

    def sample(self, **kwargs: Any) -> SampleResultType:
        raise NotImplementedError()

    def expectation(self, **kwargs: Any) -> ExpectationResultType:
        raise NotImplementedError()
