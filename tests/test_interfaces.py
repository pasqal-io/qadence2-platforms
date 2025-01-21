from __future__ import annotations

from collections import Counter

import numpy as np
import qutip
import torch
from pulser import Sequence as PulserSequence
from pulser.register import RegisterLayout
from qadence2_expressions import Z
from qadence2_ir.types import Model

from qadence2_platforms.backends.fresnel1.sequence import Fresnel1
from qadence2_platforms.backends.fresnel1.interface import Interface as Fresnel1Interface
from qadence2_platforms.backends.pyqtorch.interface import Interface as PyQInterface


N_SHOTS = 4_000
ATOL = 0.06 * N_SHOTS


def test_pyq_interface(model1: Model, pyq_interface1: PyQInterface) -> None:
    assert pyq_interface1.info == dict(num_qubits=model1.register.num_qubits)

    fparams = {"x": torch.tensor(1.0, requires_grad=True)}
    run_res = pyq_interface1.run(fparams)
    assert isinstance(run_res, torch.Tensor)
    assert run_res.shape == torch.Size([2, 2, 1])

    sample = pyq_interface1.sample(fparams, shots=N_SHOTS)[0]
    assert isinstance(sample, Counter)
    assert {"00", "11"}.issubset(set(sample.keys()))

    obs = Z(0) * Z(1)
    run_obs = pyq_interface1.expectation(fparams, shots=N_SHOTS, observable=obs)
    assert isinstance(run_obs, torch.Tensor)


def test_fresnel1_interface(
    fresnel1_register1: RegisterLayout,
    fresnel1_sequence1: PulserSequence,
    fresnel1_interface1: Fresnel1Interface,
) -> None:
    assert fresnel1_interface1.info == dict(device=Fresnel1, register=fresnel1_sequence1.register)

    fparams = {"x": 1.0}
    run_res = fresnel1_interface1.run(fparams)
    assert isinstance(run_res, qutip.Qobj)
    assert run_res.dims == [[2, 2], [1, 1]]
    assert run_res.shape == (4, 1)
    assert run_res.isket

    sample = fresnel1_interface1.sample(fparams, shots=N_SHOTS)
    assert isinstance(sample, Counter)
    assert np.allclose(sample["01"], sample["10"], atol=ATOL)

    obs = Z(0) * Z(1)
    obs_res = fresnel1_interface1.expectation(fparams, shots=N_SHOTS, observable=obs)[0]
    assert all([(0.0 <= abs(k) <= 1.0) for k in obs_res])
