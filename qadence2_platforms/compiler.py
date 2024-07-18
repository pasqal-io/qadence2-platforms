from __future__ import annotations

from types import ModuleType
from typing import Optional, Union

from qadence2_ir import Model
from qadence2.platforms.backend.bytecode import BytecodeApi
from qadence2.platforms.backend.dialect import DialectApi
from qadence2.platforms.backend.interface import RuntimeInterfaceApi
from qadence2.platforms.backend.utils import get_dialect_instance


def compile(
    model: Model,
    backend_name: str,
    device: Optional[str] = None,
    output_bytecode: bool = False,
) -> Union[RuntimeInterfaceApi, BytecodeApi]:
    """
    Compilation function to take a `Model`, a `backend`, an optional `device` and
    produce a runtime-like instance that holds the backend-specific transformed `Model`
    data. It can be used to run the given quantum program through call methods.

    It relies on the `Dialect` class that will convert all the `Model` data into
    backend-specific data. It will finally be compiled into the `RuntimeInterface` or
    `Bytecode` instance to be used for runtime purposes.

    :param model: `Model` instance containing all the data to be compiled.
    :param backend_name: `str` as a backend name.
    :param device: `str` as a device name, or `None` if no device name is provided.
    :param output_bytecode: `bool` to turn output bytecode into a `Bytecode` instance.
    :return: `RuntimeInterface` or `Bytecode` instance.
    """

    dialect_module: ModuleType = get_dialect_instance(backend_name)
    dialect: DialectApi = dialect_module.Dialect(backend_name, model, device)
    if output_bytecode:
        return dialect.compile_bytecode()
    return dialect.compile()
