from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtorch as pyq
import torch
from qadence2_ir.types import Model

from qadence2_platforms.compiler import compile_to_backend
from qadence2_platforms.utils.module_importer import resolve_module_path


def test_pyq_compilation(model1: Model) -> None:
    model = model1
    compiled_model = compile_to_backend(model, "pyqtorch")
    f_params = {"x": torch.rand(1, requires_grad=True)}
    wf = compiled_model.run(state=pyq.zero_state(2), values=f_params)
    dfdx = torch.autograd.grad(wf, f_params["x"], torch.ones_like(wf))[0]
    assert not torch.all(torch.isnan(dfdx))


# TODO: change this name for `test_qutip_compilation` when
#  other backends are available/implemented
def test_pulser_compilation(model1: Model) -> None:
    model = model1
    compiled_model = compile_to_backend(model, "fresnel1")
    f_params = {"x": np.array([1])}
    res = compiled_model.run(values=f_params)
    assert np.allclose((res * res.dag()).tr(), 1.0)


def test_custom_backend_compilation(model1: Model) -> None:
    cbk = Path(__file__).parent / "custom_backend"
    if not resolve_module_path(cbk):
        raise ValueError("Custom backend module could not be symlinked.")

    model = model1
    compiled_model = compile_to_backend(model, "custom_backend")
    f_params = {"x": np.array([1])}
    res = compiled_model.run(values=f_params)
    assert np.allclose((res * res.dag()).tr(), 1.0)
