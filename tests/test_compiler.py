from __future__ import annotations

import numpy as np
import pyqtorch as pyq
import torch
from qadence2_ir.types import Model

from qadence2_platforms.compiler import compile_to_backend


def test_pyq_compilation(model1: Model) -> None:
    model = model1
    compiled_model = compile_to_backend("pyqtorch", model)
    f_params = {"x": torch.rand(1, requires_grad=True)}
    wf = compiled_model.run(state=pyq.zero_state(2), values=f_params)
    dfdx = torch.autograd.grad(wf, f_params["x"], torch.ones_like(wf))[0]
    assert not torch.all(torch.isnan(dfdx))


# TODO: change this name for `test_qutip_compilation` when
#  other backends are available/implemented
def test_pulser_compilation(model1: Model) -> None:
    model = model1
    compiled_model = compile_to_backend("fresnel1", model)
    f_params = {"x": np.array([1])}
    res = compiled_model.run(values=f_params, on="emulator")
    assert np.allclose((res * res.dag()).tr(), 1.0)
