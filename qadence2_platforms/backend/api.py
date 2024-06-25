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
    SupportType,
    ArgsType,
    BackendInstructResultType,
)


class QuInstructAPI(
    ABC,
    Generic[SequenceObjectType, SupportType, ArgsType, BackendInstructResultType]
):
    @classmethod
    @abstractmethod
    def not_fn(
        cls,
        seq: SequenceObjectType,
        support: SupportType,
        *args: ArgsType,
        **options: Any,
    ) -> BackendInstructResultType: ...

    @classmethod
    @abstractmethod
    def z_fn(
        cls,
        seq: SequenceObjectType,
        support: SupportType,
        *args: ArgsType,
        **options: Any,
    ) -> BackendInstructResultType: ...

    @classmethod
    @abstractmethod
    def h_fn(
        cls,
        seq: SequenceObjectType,
        support: SupportType,
        *args: ArgsType,
        **options: Any,
    ) -> BackendInstructResultType: ...

    @classmethod
    @abstractmethod
    def rx_fn(
        cls,
        seq: SequenceObjectType,
        support: SupportType,
        *args: ArgsType,
        **options: Any,
    ) -> BackendInstructResultType: ...

    @classmethod
    @abstractmethod
    def qubit_dyn_fn(
        cls,
        seq: SequenceObjectType,
        support: SupportType,
        *args: ArgsType,
        **options: Any,
    ) -> BackendInstructResultType: ...


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
