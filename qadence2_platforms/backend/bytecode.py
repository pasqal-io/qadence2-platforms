from __future__ import annotations

from typing import Callable, Generic, Iterator, Optional, Union

from qadence2_platforms.qadence_ir import Alloc, Assign, Call
from qadence2_platforms.types import (
    BytecodeInstructType,
    DeviceType,
    SequenceObjectType,
)


class Bytecode(Iterator, Generic[BytecodeInstructType, SequenceObjectType]):
    """
    An iterator class to be used by the runtime function. It contains which backend
    to invoke, sequence instance and an iterable of instructions' partial functions
    to be filled with user input (if necessary) or called directly, as well as the
    device, if applicable.
    """

    def __init__(
        self,
        backend: str,
        sequence: SequenceObjectType,
        instructions: tuple[BytecodeInstructType, ...],
        variables: dict[str, Union[Call, Alloc, Assign]],
        device: DeviceType | None = None,
    ):
        self.backend: str = backend
        self.device: Optional[DeviceType] = device
        self.sequence: SequenceObjectType = sequence
        self.variables: dict[str, Union[Call, Alloc, Assign]] = variables
        self.instructions: tuple[BytecodeInstructType, ...] = instructions

    def __next__(self) -> Callable:
        raise NotImplementedError()
