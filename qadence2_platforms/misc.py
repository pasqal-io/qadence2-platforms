from __future__ import annotations

import logging
import sys
from importlib import import_module
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

BASE_BACKEND_MODULE = "qadence2_platforms.backend"


class ModuleError(Exception):
    def __init__(self, arg: Any):
        super().__init__(arg)


def module_loader(module_name: str) -> ModuleType:
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

    if module_name in sys.modules:
        return sys.modules[module_name]

    try:
        module: ModuleType = import_module(module_name)

    except ModuleNotFoundError:
        path: Path = Path(module_name)
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
            f"In case it is a custom module, please double check its path."
        )
        logger.error(error_msg)
        raise ModuleError(error_msg)

    else:
        return module
