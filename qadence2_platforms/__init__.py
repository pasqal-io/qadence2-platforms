from __future__ import annotations

from torch import float64, set_default_dtype

from .abstracts import AbstractInterface

set_default_dtype(float64)

PACKAGE_NAME = __name__
BACKEND_FOLDER_NAME = "backends"
USER_BACKENDS_FOLDER_NAME = "user_backends"
CUSTOM_BACKEND_FOLDER_NAME = "custom_backends"
TEMPLATES_FOLDER_NAME = "templates"

BASE_BACKEND_MODULE = f"{PACKAGE_NAME}.{BACKEND_FOLDER_NAME}"
USER_BACKEND_MODULE = f"{PACKAGE_NAME}.{USER_BACKENDS_FOLDER_NAME}"
