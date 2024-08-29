from __future__ import annotations

from pathlib import Path

from qadence2_platforms.utils.module_importer import resolve_module_path


def test_resolve_module() -> None:
    backend_path = Path(__file__).parent / "custom_backend"
    assert resolve_module_path(backend_path)
