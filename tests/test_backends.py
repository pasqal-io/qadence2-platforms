from __future__ import annotations

from collections import Counter

import numpy as np
import qutip
from pulser import Sequence
from pulser.register import RegisterLayout
from qadence2_expressions.operators import Z
from qadence2_ir.types import Model

from qadence2_platforms.abstracts import OnEnum
from qadence2_platforms.backends.fresnel1.interface import Interface as Fresnel1Interface
from qadence2_platforms.backends.fresnel1.sequence import Fresnel1
from qadence2_platforms.backends.pyqtorch.interface import Interface as PyQInterface


def test_pyq_interface(model1: Model, pyq_interface1: PyQInterface) -> None:
    assert pyq_interface1.info == dict(num_qubits=model1.register.num_qubits)
    # assert pyq_interface1.


def test_pyq_observables() -> None:
    pass


def test_fresnel1_interface(
    model1: Model,
    fresnel1_interface1: Fresnel1Interface,
    fresnel1_register1: RegisterLayout,
    fresnel1_sequence1: Sequence,
) -> None:
    assert fresnel1_interface1.info == dict(device=Fresnel1, register=fresnel1_sequence1.register)

    fparams = {"x": 1.0}
    sample = fresnel1_interface1.sample(fparams, shots=2_000, on=OnEnum.EMULATOR)
    assert isinstance(sample, Counter)
    assert "10" in sample and "01" in sample

    run = fresnel1_interface1.run(fparams, on=OnEnum.EMULATOR)
    assert isinstance(run, qutip.Qobj)
    assert run.dims == [[2, 2], [1, 1]]
    assert run.shape == (4, 1)
    assert run.isket

    obs = Z(0).__kron__(Z(1))
    expect = fresnel1_interface1.expectation(fparams, observable=obs)[0]
    assert np.all(k < 0.0 for k in expect)


def test_fresnel1_observables(fresnel1_interface1: Fresnel1Interface) -> None:
    pass
