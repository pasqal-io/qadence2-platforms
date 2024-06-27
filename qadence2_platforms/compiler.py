from __future__ import annotations

from types import ModuleType
from typing import Optional, Union

from qadence2_platforms import Model
from qadence2_platforms.backend.bytecode import BytecodeApi
from qadence2_platforms.backend.dialect import DialectApi
from qadence2_platforms.backend.interface import RuntimeInterfaceApi
from qadence2_platforms.backend.utils import get_dialect_instance


def compile(
    model: Model,
    backend_name: str,
    device: Optional[str] = None,
    output_bytecode: bool = False,
) -> Union[RuntimeInterfaceApi, BytecodeApi]:
    dialaect_instance: ModuleType = get_dialect_instance(backend_name)
    dialect: DialectApi = dialaect_instance.Dialect(backend_name, model, device)
    if output_bytecode:
        return dialect.compile_bytecode()
    return dialect.compile()
