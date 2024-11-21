from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pytest
import qutip
import torch
from qutip import tensor
from pulser import Sequence
from pulser.register import RegisterLayout
from qadence2_expressions.operators import Z
from qadence2_ir.types import Model

from qadence2_platforms.abstracts import OnEnum
from qadence2_platforms.backends.fresnel1.functions import (
    QuTiPObservablesParser,
    parse_native_observables as fresnel1_parse_nat_obs,
)
from qadence2_platforms.backends.fresnel1.interface import Interface as Fresnel1Interface
from qadence2_platforms.backends.fresnel1.sequence import Fresnel1
from qadence2_platforms.backends.pyqtorch.interface import Interface as PyQInterface
from qadence2_platforms.backends.pyqtorch.functions import (
    parse_native_observables as pyqtorch_parse_nat_obs,
)
from qadence2_platforms.backends.utils import InputType

N_SHOTS = 2_000
qi = qutip.qeye(2)
qz = qutip.sigmaz()


def test_pyq_interface(model1: Model, pyq_interface1: PyQInterface) -> None:
    assert pyq_interface1.info == dict(num_qubits=model1.register.num_qubits)
    fparams = {"x": torch.tensor(1.0, requires_grad=True)}
    sample = pyq_interface1.sample(fparams, shots=N_SHOTS)[0]
    assert {"00", "11"}.issubset(set(sample.keys()))


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
    sample = fresnel1_interface1.sample(fparams, shots=N_SHOTS, on=OnEnum.EMULATOR)
    assert isinstance(sample, Counter)
    assert {"10", "01"}.issubset(set(sample.keys()))

    run = fresnel1_interface1.run(fparams, on=OnEnum.EMULATOR)
    assert isinstance(run, qutip.Qobj)
    assert run.dims == [[2, 2], [1, 1]]
    assert run.shape == (4, 1)
    assert run.isket

    obs = Z(0).__kron__(Z(1))
    expect = fresnel1_interface1.expectation(fparams, observable=obs)[0]
    assert np.all(k < 0.0 for k in expect)


@pytest.mark.parametrize(
    "interface, n_qubits, expr_obs, qutip_obs",
    [
        ("fresnel1_interface", 2, Z(0), tensor(qz, qi)),
        ("fresnel1_interface", 2, Z(0).__kron__(Z(1)), tensor(qz, qz)),
        ("fresnel1_interface", 2, Z(0) + Z(1), tensor(qz, qi) + tensor(qi, qz)),
    ],
)
def test_fresnel1_observables(
    interface: Fresnel1Interface,
    n_qubits: int,
    expr_obs: InputType,
    qutip_obs: qutip.Qobj,
    request: Any,
) -> None:
    parsed_obs = QuTiPObservablesParser.build(num_qubits=n_qubits, observables=expr_obs)
    assert np.allclose(parsed_obs, qutip_obs)
    assert np.allclose(parsed_obs, fresnel1_parse_nat_obs(n_qubits, expr_obs))
