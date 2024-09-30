from __future__ import annotations

import os
from pathlib import Path

from qadence2_platforms.utils import BackendTemplate

BACKEND_NAME = "custom_backend_test"
BACKEND_PATH = Path(__file__).parent


def test_create_custom_backend() -> None:
    template = BackendTemplate()
    assert template.create_template(BACKEND_NAME, gui=False, use_this_dir=BACKEND_PATH)
    assert os.path.exists(template.user_backend_path)
    assert all(
        os.path.exists(template.user_backend_path / k)
        for k in template.template_files_list
    )
    assert os.path.exists(template.platforms_backend_path)
