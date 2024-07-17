from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic

from qadence2_platforms.types import (
    EmbeddingType,
    ExpectationResultType,
    InterfaceCallResultType,
    NativeBackendType,
    NativeSequenceType,
    RegisterType,
    RunResultType,
    SampleResultType,
)


class RuntimeInterfaceApi(
    ABC,
    Generic[
        RegisterType,
        EmbeddingType,
        NativeSequenceType,
        NativeBackendType,
        RunResultType,
        SampleResultType,
        ExpectationResultType,
        InterfaceCallResultType,
    ],
):
    """
    Interface generic class to be used to build runtime classes for backends.
    It may run with the qadence-core runtime functions when post-processing,
    statistical analysis, etc.
    """

    register: RegisterType
    embedding: EmbeddingType
    engine: NativeBackendType
    sequence: NativeSequenceType

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> InterfaceCallResultType:
        raise NotImplementedError()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> InterfaceCallResultType:
        raise NotImplementedError()

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> RunResultType:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> SampleResultType:
        raise NotImplementedError()

    @abstractmethod
    def expectation(self, *args: Any, **kwargs: Any) -> ExpectationResultType:
        raise NotImplementedError()
