from __future__ import annotations

from pathlib import Path

from qadence2_platforms.utils import BackendTemplate

BACKEND_NAME = "custom_backend_test"
BACKEND_PATH = Path(__file__).parent


def test_create_custom_backend() -> None:
    assert BackendTemplate().create_template(
        BACKEND_NAME, gui=False, use_this_dir=BACKEND_PATH
    )
