from __future__ import annotations

from typing import Callable

from qadence2_platforms.backend.bytecode import BytecodeApi


class Bytecode(BytecodeApi):
    def __next__(self) -> Callable:
        raise NotImplementedError()
