from __future__ import annotations

import numpy as np
from collections import Counter

from qadence2_platforms.qadence_ir import (
    Model,
    AllocQubits,
    Alloc,
    Call,
    Load,
    Assign,
    QuInstruct,
    Support,
)
from qadence2_platforms.compiler import compile
from qadence2_platforms.types import DeviceName
from qadence2_platforms.backend.pulser.interface import RuntimeInterface


def test_pulser_fresnel_eom_compilation(model: Model) -> None:
    compiled_model = compile(model, "pulser", DeviceName.FRESNEL_EOM)
    assert isinstance(compiled_model, RuntimeInterface)
    result = compiled_model.run(num_shots=1000)
    assert sum(result.values()) == 1000
