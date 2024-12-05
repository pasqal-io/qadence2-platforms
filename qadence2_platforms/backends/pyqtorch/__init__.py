from __future__ import annotations

from torch import float64, set_default_dtype

from .compiler import compile_to_backend

set_default_dtype(float64)
