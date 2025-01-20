from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import torch

from qadence2_expressions import compile_to_model, parameter, RX, reset_ir_options, Expression

from pyqtorch.utils import OrderedCounter
from qadence2_ir.types import Model

from qadence2_platforms.compiler import compile_to_backend
from qadence2_platforms.backends.fresnel1 import compile_to_backend as fresnel1_compile
from qadence2_platforms.backends.fresnel1.interface import Interface as Fresnel1Interface
from qadence2_platforms.backends.pyqtorch import compile_to_backend as pyq_compile
from qadence2_platforms.backends.pyqtorch.interface import Interface as PyQInterface

N_SHOTS = 2_000
ATOL = 0.05 * N_SHOTS


def test_pyq_compiler(model1: Model, pyq_interface1: PyQInterface) -> None:
    interface = pyq_compile(model1)
    assert pyq_interface1.info == interface.info
    assert pyq_interface1.sequence.qubit_support == interface.sequence.qubit_support
    assert [
        p1 == p2
        for p1, p2 in zip(pyq_interface1.sequence.operations, interface.sequence.operations)
    ]


@pytest.mark.parametrize(
    "operator, angle, values", [(RX, 2.0, {}), (RX, parameter("x"), {"x": torch.rand(1)})]
)
def test_pyq_expr_compilation(operator: Callable, angle: float | Expression, values: dict) -> None:
    reset_ir_options()
    model = compile_to_model(operator(angle)(0))
    bkd = compile_to_backend(model, "pyqtorch")
    (sample,) = bkd.sample(values=values, shots=2000)
    assert isinstance(sample, OrderedCounter)
    reset_ir_options()


def test_fresnel1_compiler(model1: Model, fresnel1_interface1: Fresnel1Interface) -> None:
    interface = fresnel1_compile(model1)
    fparams = {"x": 1.0}
    assert fresnel1_interface1.info == interface.info
    assert fresnel1_interface1.sequence.register == interface.sequence.register
    assert fresnel1_interface1.sequence.device == interface.sequence.device
    assert fresnel1_interface1.parameters() == interface.parameters()
    assert np.allclose(
        np.array(list(fresnel1_interface1.sample(fparams, shots=N_SHOTS).values())),
        np.array(list(interface.sample(fparams, shots=N_SHOTS).values())),
        atol=ATOL,
    )
