from __future__ import annotations

from qadence2_ir.types import Model

from qadence2_platforms.backends.fresnel1 import compile_to_backend as fresnel1_compile
from qadence2_platforms.backends.fresnel1.interface import Interface as Fresnel1Interface
from qadence2_platforms.backends.pyqtorch import compile_to_backend as pyq_compile
from qadence2_platforms.backends.pyqtorch.interface import Interface as PyQInterface


def test_pyq_compiler(model1: Model, pyq_interface1: PyQInterface) -> None:
    interface = pyq_compile(model1)
    assert pyq_interface1.info == interface.info
    assert pyq_interface1.sequence.qubit_support == interface.sequence.qubit_support
    assert [p1 == p2 for p1, p2 in zip(pyq_interface1.sequence.operations, interface.sequence.operations)]


def test_fresnel1_compiler(model1: Model, fresnel1_interface1: Fresnel1Interface) -> None:
    interface = fresnel1_compile(model1)
    assert fresnel1_interface1.info == interface.info
    assert fresnel1_interface1.sequence.register == interface.sequence.register
    assert fresnel1_interface1.sequence.device == interface.sequence.device
    assert fresnel1_interface1.parameters() == interface.parameters()
