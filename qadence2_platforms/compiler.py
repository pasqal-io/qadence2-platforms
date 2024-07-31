from __future__ import annotations

from importlib import import_module
from qadence2_ir.types import Model

from .abstracts import AbstractInterface as Interface


def compile_to_backend(platform: str, model: Model) -> Interface:
    plat = import_module(f"qadence2_platforms.backend.{platform}")
    return plat.compile_to_backend(model)
