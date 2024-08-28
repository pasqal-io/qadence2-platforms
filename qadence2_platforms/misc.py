from __future__ import annotations

import logging
import os
import sys
from importlib import import_module
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

from qadence2_platforms import BACKEND_FOLDER_NAME, BASE_BACKEND_MODULE

logger = logging.getLogger(__name__)


class ModuleError(Exception):
    def __init__(self, arg: Any):
        super().__init__(arg)


def full_backend_path(backend: str) -> Path:
    folder = Path(__file__).parent
    return folder / BACKEND_FOLDER_NAME / backend


def check_backend_exists(backend: Path) -> bool:
    # folder = Path(__file__).parent
    # path = Path(folder, BACKEND_FOLDER_NAME, backend)
    # return os.path.exists(os.path.dirname(path))
    return os.path.exists(backend)


def module_loader(module_name: str | Path) -> ModuleType:
    """
    Loads an arbitrary module and returns it. It can be a submodule from an already
    imported module, i.e. "torch.nn", an existing but not imported module, i.e.
    "scipy", or a python file path containing a module, i.e.
    "/Users/user/dir/some_module.py".

    It facilitates to import and work with custom modules for backend development,
    for instance.

    :param module_name: str: The name of the module to load. It may be its name,
        a full module name, or a file path.
    :return: ModuleType: The loaded module.
    """

    if isinstance(module_name, Path):
        return _path_module_loader(module_name)
    return _str_module_loader(module_name)


def _str_module_loader(module_name: str) -> ModuleType:
    str_module_name = f"{BASE_BACKEND_MODULE}.{module_name}"
    if str_module_name in sys.modules:
        return sys.modules[str_module_name]

    try:
        module: ModuleType = import_module(str_module_name)

    except ModuleNotFoundError:
        path: Path = Path(str_module_name)
        file: str = path.name
        spec: ModuleSpec | None = spec_from_file_location(file, path)
        if spec is not None:
            ext_module: ModuleType = module_from_spec(spec)
            loader: Any = spec.loader
            if loader is not None:
                loader.exec_module(ext_module)
            return ext_module

        error_msg = (
            f"Module error. Please verify module '{module_name}'. "
            f"You may need to import it beforehand. "
            f"In case it is a custom module, please double check its path and"
            f"use `pathlib.Path` class instead of pure `str`."
        )
        logger.error(error_msg)
        raise ModuleError(error_msg)

    else:
        return module


def _path_module_loader(module_name: Path) -> ModuleType:
    path: Path = Path(module_name)
    file: str = path.name
    print(path, file)
    spec: ModuleSpec | None = spec_from_file_location(file, path)
    if spec is not None:
        ext_module: ModuleType = module_from_spec(spec)
        loader: Any = spec.loader
        if loader is not None:
            loader.exec_module(ext_module)
        return ext_module
    raise ValueError("?")
