from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Callable


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


def get_register_instance(backend: str) -> Callable:
    module_name = f"qadence2_platforms.backend.{backend}.backend"
    return getattr(import_module(name=module_name), "get_backend_register")
