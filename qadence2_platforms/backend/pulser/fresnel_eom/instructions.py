from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
from pulser.sequence import Sequence

from qadence2_platforms.qadence_ir import Support

from ..backend import BackendInstructResult
from .functions import free_evolution, h_pulse, rotation


def not_fn(
    seq: Sequence,
    support: Support,
    *args: Any,
    **_: Any,
) -> BackendInstructResult:
    return BackendInstructResult(
        fn=partial(rotation, sequence=seq, support=support, angle=np.pi, direction="x"),
        *args,
    )


def h_fn(
    seq: Sequence, support: Support, *args: Any, **_: Any
) -> BackendInstructResult:
    return BackendInstructResult(
        fn=partial(h_pulse, sequence=seq, support=support), *args
    )


def rx_fn(
    seq: Sequence, support: Support, *args: Any, **_: Any
) -> BackendInstructResult:
    return BackendInstructResult(
        fn=partial(rotation, sequence=seq, support=support, direction="x"), *args
    )


def qubit_dyn_fn(
    seq: Sequence, support: Support, *args: Any, **_: Any
) -> BackendInstructResult:
    # for the sake of testing purposes, qubit dynamics will be only
    # a simple free evolution pulse
    return BackendInstructResult(
        fn=partial(free_evolution, sequence=seq, support=support), *args
    )
