from __future__ import annotations

from qadence2_ir.types import Model

from qadence2_platforms.backends.fresnel1.interface import Interface as Fresnel1Interface
from qadence2_platforms.backends.pyqtorch.interface import Interface as PyQInterface


def test_interface(
    model1: Model, pyq_interface1: PyQInterface, fresnel1_interface1: Fresnel1Interface
) -> None:
    assert pyq_interface1.info == dict(num_qubits=model1.register.num_qubits)
    # assert pyq_interface1.
