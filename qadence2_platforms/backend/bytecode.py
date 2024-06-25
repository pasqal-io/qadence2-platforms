from __future__ import annotations

from types import ModuleType
from typing import Any, Callable, Iterator, Generic, Optional, Union

from qadence2_platforms.types import (
    DeviceType,
    BytecodeInstructType,
    SequenceObjectType,
)
from qadence2_platforms.qadence_ir import Alloc, Assign


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
        instructions: BytecodeInstructType,
        variables: dict[str, Union[Alloc, Assign]],
        device: DeviceType | None = None
    ):
        self.backend: str = backend
        self.device: Optional[DeviceType] = device
        self.sequence: SequenceObjectType = sequence
        self.variables: dict[str, Union[Alloc, Assign]] = variables
        self.instructions: BytecodeInstructType = instructions

    def __next__(self) -> Callable:
        ...
