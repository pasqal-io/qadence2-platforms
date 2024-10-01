from __future__ import annotations

from typing import cast

from qadence2_ir.types import Model

from qadence2_platforms.utils.module_importer import module_loader

from .abstracts import AbstractInterface as Interface


def compile_to_backend(model: Model, backend: str) -> Interface:
    """
    Function that gets a `Model` (Qadence IR) and a backend name, and.

    returns an `Interface` instance from the specific backend with the
    model transformed into backend appropriate data.

    :param model: (Model) qadence IR
    :param backend: (str) the backend to be used to execute the Model
    :return: (Interface) interface instance of the chosen backend
    """

    plat = module_loader(backend)
    return cast(Interface, plat.compile_to_backend(model))
