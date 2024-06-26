from __future__ import annotations

import pyqtorch as pyq

from qadence2_platforms.backend.pyqtorch.parse import parse_instruction
from qadence2_platforms.qadence_ir import (
    Alloc,
    AllocQubits,
    Assign,
    Call,
    Load,
    Model,
    QuInstruct,
    Support,
)


def test_call_conversion() -> None:
    pass


def test_pyq_conversion() -> None:
    model = Model(
        register=AllocQubits(num_qubits=2, options={"initial_state": "10"}),
        inputs={
            "x": Alloc(size=1, trainable=False),
        },
        instructions=[
            # -- Feature map
            Assign("%0", Call("mul", 1.57, Load("x"))),
            Assign("%1", Call("sin", Load("%0"))),
            QuInstruct("rx", Support(target=(0,)), Load("%1")),
            # --
            QuInstruct("not", Support(target=(1,), control=(0,))),
        ],
        directives={"digital": True},
        data_settings={"result-type": "state-vector", "data-type": "f32"},
    )
    Assign("%0", Call("mul", 1.57, Load("x")))
    native_operations = parse_instruction(model.instructions)
    pyq_circuit = pyq.QuantumCircuit(model.register.num_qubits, native_operations)


if __name__ == "__main__":
    test_pyq_conversion()