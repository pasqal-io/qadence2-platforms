from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Callable, Generic

from qadence2_platforms.types import (
    BytecodeInstructType,
    UserInputType,
    DeviceType,
)
from qadence2_platforms.qadence_ir import Model


def get_backend_module(backend: str) -> ModuleType:
    module_name = f"qadence2_platforms.backend.{backend}"
    return import_module(name=module_name, package="backend")


def get_device_module(backend: str, device: str) -> ModuleType:
    module_name = f"qadence2_platforms.backend.{backend}.backend"
    return import_module(name=module_name, package=device)


def get_backend_instruct_instance(backend: str, device: str) -> Callable:
    module_name = f"qadence2_platforms.backend.{backend}"
    if device:
        module_name += f".{device}"
    return getattr(import_module(name=module_name, package="instructions"), "BackendInstruct")


def get_sequence_instance(backend: str, device: str) -> Callable:
    module_name = f"qadence2_platforms.backend.{backend}"
    if device:
        module_name += f".{device}"
    return getattr(import_module(name=module_name, package="instructions"), "BackendSequence")


def get_backend_register_fn(backend: str) -> Callable:
    module_name = f"qadence2_platforms.backend.{backend}.backend"
    return getattr(import_module(name=module_name), "get_backend_register")


class BackendInstructResult(Generic[UserInputType, BytecodeInstructType]):
    def __init__(self, fn: Callable, *args: Any):
        self._fn = fn
        self._args = args

    @property
    def fn(self) -> Callable:
        return self._fn

    @property
    def args(self) -> tuple[Any, ...]:
        return self._args

    def resolve_args(self, inputs: UserInputType) -> BytecodeInstructType:
        pass
