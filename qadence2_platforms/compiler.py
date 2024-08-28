from __future__ import annotations

from qadence2_ir.types import Model

from qadence2_platforms.utils.module_importer import module_loader

from .abstracts import AbstractInterface as Interface


def compile_to_backend(backend: str, model: Model) -> Interface:
    plat = module_loader(backend)
    return plat.compile_to_backend(model)
