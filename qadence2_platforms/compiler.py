from __future__ import annotations

from pathlib import Path

from qadence2_ir.types import Model

from .abstracts import AbstractInterface as Interface
from .misc import module_loader


def compile_to_backend(backend: str | Path, model: Model) -> Interface:
    plat = module_loader(backend)
    return plat.compile_to_backend(model)
