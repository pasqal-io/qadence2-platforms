from __future__ import annotations

from .abstracts import AbstractInterface, OnEnum

PACKAGE_NAME = __name__
BACKEND_FOLDER_NAME = "backends"
USER_BACKENDS_FOLDER_NAME = "user_backends"
CUSTOM_BACKEND_FOLDER_NAME = "custom_backends"
TEMPLATES_FOLDER_NAME = "templates"

BASE_BACKEND_MODULE = f"{PACKAGE_NAME}.{BACKEND_FOLDER_NAME}"
USER_BACKEND_MODULE = f"{PACKAGE_NAME}.{USER_BACKENDS_FOLDER_NAME}"

__all__ = ["AbstractInterface", "OnEnum", "BASE_BACKEND_MODULE", "USER_BACKEND_MODULE"]
