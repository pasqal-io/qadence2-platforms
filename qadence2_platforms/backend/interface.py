from __future__ import annotations

from typing import Any, Generic
from abc import ABC, abstractmethod

from qadence2_platforms.backend.bytecode import BytecodeApi
from qadence2_platforms.backend.utils import get_backend_module
from qadence2_platforms.types import (
    ExpectationResultType,
    InterfaceCallResultType,
    InterfaceInstructType,
    RunResultType,
    SampleResultType,
)


class RuntimeInterface(
    ABC,
    Generic[
        InterfaceInstructType,
        RunResultType,
        SampleResultType,
        ExpectationResultType,
        InterfaceCallResultType,
    ]
):
    """
    Interface generic class to be used to build runtime classes for backends.
    It may run with the qadence-core runtime functions when post-processing,
    statistical analysis, etc.
    """
    def __init__(self, bytecode: BytecodeApi, **options: Any):
        self.bytecode: BytecodeApi = bytecode
        self.options: Any = options

    def _resolve_parameters(self) -> Any:
        resolve_fn = getattr(
            get_backend_module(self.bytecode.backend), "resolve_parameters"
        )
        return resolve_fn(
            instructions=self.bytecode.instructions, variables=self.bytecode.variables
        )

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> InterfaceCallResultType:
        raise NotImplementedError()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> InterfaceCallResultType:
        raise NotImplementedError()

    @abstractmethod
    def run(self, **kwargs: Any) -> RunResultType:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, **kwargs: Any) -> SampleResultType:
        raise NotImplementedError()

    @abstractmethod
    def expectation(self, **kwargs: Any) -> ExpectationResultType:
        raise NotImplementedError()
