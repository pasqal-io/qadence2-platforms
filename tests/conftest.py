# use this file for configuring test fixtures and
# functions common to every test
from __future__ import annotations

from pyqtorch import QuantumCircuit
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

from qadence2_platforms.backends.fresnel1 import (
    register as fresnel1_register,
)
from qadence2_platforms.backends.fresnel1 import (
    sequence as fresnel1_sequence,
)
from qadence2_platforms.backends.fresnel1.interface import Interface as Fresnel1Interface
from qadence2_platforms.backends.pyqtorch.compiler import Compiler, get_trainable_params
from qadence2_platforms.backends.pyqtorch.embedding import Embedding
from qadence2_platforms.backends.pyqtorch.interface import Interface as PyQInterface
from qadence2_platforms.backends.pyqtorch.register import RegisterInterface


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


@fixture
def pyq_circuit1() -> QuantumCircuit:
    # return QuantumCircuit(2, [RX(0, torch.sin(1.5 * ))])
    pass


@fixture
def pyq_interface1(model1: Model) -> PyQInterface:
    register = RegisterInterface(
        n_qubits=model1.register.num_qubits,
        init_state=model1.register.options.get("initial_state"),
    )
    embedding = Embedding(model1)
    circuit = Compiler().compile(model1)
    vparams = get_trainable_params(model1.inputs)

    return PyQInterface(
        register=register,
        embedding=embedding,
        circuit=circuit,
        vparams=vparams,
    )


@fixture
def fresnel1_interface1(model1: Model) -> Fresnel1Interface:
    register = fresnel1_register.from_model(model1)
    sequence = fresnel1_sequence.from_model(model1, register)
    fparams = {k for k, v in model1.inputs.items() if not v.is_trainable}

    return Fresnel1Interface(sequence, fparams)
