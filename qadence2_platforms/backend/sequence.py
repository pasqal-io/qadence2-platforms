from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional

from qadence2_platforms import Model
from qadence2_platforms.types import NativeSequenceType, DeviceType, RegisterType


class SequenceApi(ABC, Generic[NativeSequenceType]):
    """
    A generic sequence api class to produce the native sequence of instructions
    for the backend.
    """

    model: Model
    register: RegisterType
    device: Optional[DeviceType]
    instruction_map: dict[str, Callable]

    @abstractmethod
    def build_sequence(self) -> NativeSequenceType:
        raise NotImplementedError()
