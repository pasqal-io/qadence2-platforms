from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Callable, Generic, Iterator, Union

from qadence2_platforms.types import (
    BytecodeInstructType,
    UserInputType,
)


def get_backend_module(backend: str) -> ModuleType:
    module_name = f"qadence2_platforms.backend.{backend}"
    return import_module(name=module_name, package="backend")


def get_device_module(backend: str, device: str) -> Any:
    module_name = f"qadence2_platforms.backend.{backend}.backend"
    return getattr(import_module(name=module_name), device)


def get_backend_instruct_instance(backend: str, device: str) -> Any:
    module_name = f"qadence2_platforms.backend.{backend}"
    if device:
        module_name += f".{device}"
    return getattr(
        import_module(name=module_name, package="instructions"), "BackendInstruct"
    )


def get_native_seq_instance(backend: str, device: str) -> Any:
    module_name = f"qadence2_platforms.backend.{backend}"
    if device:
        module_name += f".{device}"
    return getattr(
        import_module(name=module_name, package="instructions"), "BackendSequence"
    )


def get_backend_register_fn(backend: str) -> Any:
    module_name = f"qadence2_platforms.backend.{backend}.backend"
    return getattr(import_module(name=module_name), "get_backend_register")


class BackendInstructResult(Generic[UserInputType, BytecodeInstructType]):
    def __init__(self, *args: Any, fn: Union[Callable, None] = None):
        if not fn:
            raise ValueError("Must declare `fn` argument.")
        self._fn = fn
        self._args = args

    @property
    def fn(self) -> Callable:
        return self._fn

    @property
    def args(self) -> tuple[Any, ...]:
        return self._args
