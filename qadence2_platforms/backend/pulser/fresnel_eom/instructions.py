from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
from pulser.sequence import Sequence

from qadence2_platforms.qadence_ir import Support

from ..backend import InstructPartialResult
from .functions import free_evolution, h_pulse, rotation


def not_fn(
    seq: Sequence,
    support: Support,
    params: Any,
) -> InstructPartialResult:
    return InstructPartialResult(
        fn=partial(rotation, sequence=seq, support=support, angle=np.pi, direction="x"),
        params=params,
    )


def h_fn(seq: Sequence, support: Support, params: Any) -> InstructPartialResult:
    return InstructPartialResult(
        fn=partial(h_pulse, sequence=seq, support=support), params=params
    )


def rx_fn(seq: Sequence, support: Support, params: Any) -> InstructPartialResult:
    return InstructPartialResult(
        fn=partial(rotation, sequence=seq, support=support, direction="x"),
        params=params,
    )


def qubit_dyn_fn(seq: Sequence, support: Support, params: Any) -> InstructPartialResult:
    # for the sake of testing purposes, qubit dynamics will be only
    # a simple free evolution pulse
    return InstructPartialResult(
        fn=partial(free_evolution, sequence=seq, support=support), params=params
    )
