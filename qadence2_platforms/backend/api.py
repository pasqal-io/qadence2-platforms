from __future__ import annotations

from typing import Any, Callable, Generic
from abc import ABC, abstractmethod

from qadence2_platforms.types import (
    RegisterType,
    DeviceType,
    DirectivesType,
    BackendType,
    QuInstructType,
    BytecodeInstructType,
    SequenceObjectType,
)


class QuInstructAPI(ABC, Generic[QuInstructType]):
    @classmethod
    @abstractmethod
    def not_fn(cls, seq: QuInstructType, **options: Any) -> Callable: ...

    @classmethod
    @abstractmethod
    def z_fn(cls, seq: QuInstructType, **options: Any) -> Callable: ...

    @classmethod
    @abstractmethod
    def h_fn(cls, seq: QuInstructType, **options: Any) -> Callable: ...

    @classmethod
    @abstractmethod
    def rx_fn(cls, seq: QuInstructType, **options: Any) -> Callable: ...

    @classmethod
    @abstractmethod
    def qubit_dyn_fn(cls, seq: QuInstructType, **options: Any) -> Callable: ...


class BackendSequenceAPI(
    ABC,
    Generic[RegisterType, DeviceType, DirectivesType, SequenceObjectType]
):
    @abstractmethod
    def get_sequence(self) -> SequenceObjectType: ...


class DialectAPI(
    ABC,
    Generic[RegisterType, DeviceType, BackendType, BytecodeInstructType, SequenceObjectType]
):
    pass
