from __future__ import annotations

from typing import Union, Optional

from qadence2_platforms import Model
from qadence2_platforms.backend.interface import RuntimeInterfaceApi
from qadence2_platforms.backend.bytecode import BytecodeApi
from qadence2_platforms.backend.utils import get_backend_module
from qadence2_platforms.backend.dialect import DialectApi


def compile(
    model: Model,
    backend_name: str,
    device: Optional[str] = None,
    output_bytecode: bool = False,
) -> Union[RuntimeInterfaceApi, BytecodeApi]:
    backend = get_backend_module(backend_name)
    dialect: DialectApi = backend.Dialect(backend_name, model, device)
    if output_bytecode:
        return dialect.compile_bytecode()
    return dialect.compile()
