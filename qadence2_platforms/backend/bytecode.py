from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Iterator, Optional

from qadence2.platforms.types import (
    BytecodeInstructType,
    DeviceType,
    EmbeddingType,
    InstructionsObjectType,
)


class BytecodeApi(
    Iterator, Generic[BytecodeInstructType, EmbeddingType, InstructionsObjectType], ABC
):
    """
    An iterator class to be used by the runtime function. It contains which backend
    to invoke, sequence instance and an iterable of instructions' partial functions
    to be filled with user input (if necessary) or called directly, as well as the
    device, if applicable.
    """

    def __init__(
        self,
        backend: str,
        sequence: InstructionsObjectType,
        instructions: BytecodeInstructType,
        variables: EmbeddingType,
        device: DeviceType | None = None,
    ):
        self.backend: str = backend
        self.device: Optional[DeviceType] = device
        self.sequence: InstructionsObjectType = sequence
        self.variables: EmbeddingType = variables
        self.instructions: BytecodeInstructType = instructions

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Callable:
        raise NotImplementedError()

    @abstractmethod
    def __next__(self) -> Callable:
        raise NotImplementedError()
