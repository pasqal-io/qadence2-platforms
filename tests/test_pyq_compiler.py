from __future__ import annotations

import pyqtorch as pyq
import torch

from qadence2_platforms.compiler import compile_to_backend
from qadence2_ir.types import (
    Alloc,
    AllocQubits,
    Assign,
    Call,
    Load,
    Model,
    QuInstruct,
    Support,
)


def test_pyq_compilation() -> None:
    model = Model(
        register=AllocQubits(num_qubits=2, options={"initial_state": "10"}),
        inputs={
            "x": Alloc(size=1, trainable=False),
        },
        instructions=[
            Assign("%0", Call("mul", 1.57, Load("x"))),
            Assign("%1", Call("sin", Load("%0"))),
            QuInstruct("rx", Support(target=(0,)), Load("%1")),
            QuInstruct("not", Support(target=(1,), control=(0,))),
        ],
        directives={"digital": True},
        # data_settings={"result-type": "state-vector", "data-type": "f32"},
    )
    compiled_model = compile_to_backend("pyqtorch", model)
    f_params = {"x": torch.rand(1, requires_grad=True)}
    wf = compiled_model.run(state=pyq.zero_state(2), values=f_params)
    dfdx = torch.autograd.grad(wf, f_params["x"], torch.ones_like(wf))[0]
    assert not torch.all(torch.isnan(dfdx))
