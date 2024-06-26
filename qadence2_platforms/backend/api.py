from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic

from qadence2_platforms.types import (
    ArgsType,
    BackendInstructResultType,
    DeviceType,
    DirectivesType,
    RegisterType,
    SequenceObjectType,
    SupportType,
)


class QuInstructAPI(
    ABC, Generic[SequenceObjectType, SupportType, ArgsType, BackendInstructResultType]
):
    @classmethod
    @abstractmethod
    def not_fn(
        cls,
        seq: SequenceObjectType,
        support: SupportType,
        *args: Any,
        **options: Any,
    ) -> BackendInstructResultType: ...

    @classmethod
    @abstractmethod
    def z_fn(
        cls,
        seq: SequenceObjectType,
        support: SupportType,
        *args: Any,
        **options: Any,
    ) -> BackendInstructResultType: ...

    @classmethod
    @abstractmethod
    def h_fn(
        cls,
        seq: SequenceObjectType,
        support: SupportType,
        *args: Any,
        **options: Any,
    ) -> BackendInstructResultType: ...

    @classmethod
    @abstractmethod
    def rx_fn(
        cls,
        seq: SequenceObjectType,
        support: SupportType,
        *args: Any,
        **options: Any,
    ) -> BackendInstructResultType: ...

    @classmethod
    @abstractmethod
    def qubit_dyn_fn(
        cls,
        seq: SequenceObjectType,
        support: SupportType,
        *args: Any,
        **options: Any,
    ) -> BackendInstructResultType: ...


class BackendSequenceAPI(
    ABC, Generic[RegisterType, DeviceType, DirectivesType, SequenceObjectType]
):
    @abstractmethod
    def get_sequence(self) -> SequenceObjectType: ...
