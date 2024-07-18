from __future__ import annotations

import pyqtorch as pyq
import torch

from qadence2.platforms import Model
from qadence2.platforms.backend.api import compile
from qadence2.platforms.compiler import compile as new_compile


def test_pyq_compilation(model: Model) -> None:
    compiled_model = compile(model, "pyqtorch")
    f_params = {"x": torch.rand(1, requires_grad=True)}
    wf = compiled_model(pyq.zero_state(2), f_params)
    dfdx = torch.autograd.grad(wf, f_params["x"], torch.ones_like(wf))[0]
    assert not torch.all(torch.isnan(dfdx))


def test_new_pyq_compilation(model: Model) -> None:
    compiled_model = new_compile(model, "pyqtorch")
    f_params = {"x": torch.rand(1, requires_grad=True)}
    wf = compiled_model(pyq.zero_state(2), f_params)
    dfdx = torch.autograd.grad(wf, f_params["x"], torch.ones_like(wf))[0]
    assert not torch.all(torch.isnan(dfdx))
