from __future__ import annotations

from qadence2_ir.types import Model

from .abstracts import AbstractInterface as Interface
from .misc import BASE_BACKEND_MODULE, module_loader


def compile_to_backend(backend: str, model: Model) -> Interface:
    plat = module_loader(f"{BASE_BACKEND_MODULE}.{backend}")
    return plat.compile_to_backend(model)
