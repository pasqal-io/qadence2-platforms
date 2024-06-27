from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

from qadence2_platforms.types import device2backend_map


def get_backend_module(backend: str) -> ModuleType:
    module_name = f"qadence2_platforms.backend.{backend}"
    return import_module(name=module_name, package="backend")


def get_backend_device(backend: str, device: str) -> Any:
    module_name = f"qadence2_platforms.backend.{backend}.{device2backend_map[device]}"
    return import_module(name=module_name, package="backend")


def get_device_instance(backend: str, device: str) -> Any:
    module_name = f"qadence2_platforms.backend.{backend}.backend"
    return getattr(import_module(name=module_name), device, None)


def get_embedding_instance(backend: str) -> Any:
    module_name = f"qadence2_platforms.backend.{backend}.embedding"
    return getattr(import_module(name=module_name), "EmbeddingModule")


def get_native_seq_instance(backend: str, device: str) -> Any:
    module_name = f"qadence2_platforms.backend.{backend}"
    if device:
        module_name += f".{device2backend_map[device]}"
    return getattr(import_module(name=module_name), "Sequence")


def get_backend_register_fn(backend: str) -> Any:
    module_name = f"qadence2_platforms.backend.{backend}.backend"
    return getattr(import_module(name=module_name), "get_backend_register")
