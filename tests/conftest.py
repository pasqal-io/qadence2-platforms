# use this file for configuring test fixtures and
# functions common to every test
from __future__ import annotations

from pytest import fixture
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


@fixture
def model1() -> Model:
    return Model(
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
    )
