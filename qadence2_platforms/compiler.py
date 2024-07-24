from __future__ import annotations

from importlib import import_module
from qadence2_ir.types import Model

from .abstracts import AbstractInterface as Interface


def modelc(model: Model, platform: str) -> Interface:
    plat = import_module(f"qadence2_platforms.backend.{platform}")
    return plat.modelc(model)
