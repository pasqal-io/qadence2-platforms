from __future__ import annotations

import logging
import sys
import traceback
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import Any

from qadence2_platforms import (
    BASE_BACKEND_MODULE,
    USER_BACKEND_MODULE,
    USER_BACKENDS_FOLDER_NAME,
)

logger = logging.getLogger(__name__)


class ModuleError(Exception):
    def __init__(self, arg: Any):
        super().__init__(arg)


def module_loader(module_name: str) -> ModuleType:
    """
    Loads an arbitrary module and returns it.

    It can be a backend submodule from
    an already imported backend, i.e. `"pyqtorch"`, an existing but not imported
    module, i.e. `"fresnel1"`, or a custom backend module, i.e. `"custom_backend1"`.

    It facilitates to import and work with custom modules for backends development,
    for instance.

    Args:
        module_name (str): The name of the module to load.

    Returns:
        The loaded module.
    """

    base_backend = f"{BASE_BACKEND_MODULE}.{module_name}"
    if base_backend in sys.modules:
        return sys.modules[base_backend]
    user_backend = f"{USER_BACKEND_MODULE}.{module_name}"
    if user_backend in sys.modules:
        return sys.modules[user_backend]

    module: ModuleType
    try:
        module = import_module(base_backend)
    except ModuleNotFoundError:
        try:
            module = import_module(user_backend)
        except ModuleNotFoundError:
            traceback.print_exc()
            error_msg = (
                f"Module error. Please verify module '{module_name}'. "
                f"You may need to import it beforehand. "
                f"In case it is a custom module, please double check its path and"
                f"use `pathlib.Path` class instead of pure `str`."
            )
            logger.error(ModuleError(error_msg))
            raise ModuleError(error_msg)
    return module


def resolve_module_path(module_source: str | Path) -> bool:
    """
    Resolve module path for custom backends.

    It symlinks custom backends,
    if they are not symlinked yet, and ensure that relative imports from
    their files do not break.

    Args:
        module_source (str | Path): module source path.

    Returns:
        True if the module path is resolved, False otherwise.
    """

    platforms_spec = find_spec("qadence2_platforms")
    if platforms_spec is None:
        return False

    if platforms_spec.origin is None:
        return False

    platforms_path = Path(platforms_spec.origin).parent
    src = Path(module_source).resolve()

    try:
        dst = platforms_path / USER_BACKENDS_FOLDER_NAME
        dst.symlink_to(src, target_is_directory=True)
    except FileExistsError:
        return True
    except Exception:
        traceback.print_exc()
        return False
    return True
