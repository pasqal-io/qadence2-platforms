from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import qutip
from pulser import Sequence as PulserSequence, AnalogDevice, Register
from pulser.register import RegisterLayout
from qadence2_expressions import Z
from qadence2_ir.types import Model
from qutip import tensor as qtensor
import pyqtorch as pyq
from pyqtorch import (
    Observable as PyQObservable,
    Z as PZ,
)

from qadence2_platforms.backends.fresnel1.functions import (
    local_pulse,
    local_pulse_core,
    apply_local_shifts,
    dyn_pulse,
    dyn_wait,
    rotation,
    h,
    x,
    rx,
    ry,
    Direction,
    Duration,
    QuTiPObservablesParser,
    parse_native_observables as fresnel1_parse_nat_obs,
)
from qadence2_platforms.backends.fresnel1.interface import Interface as Fresnel1Interface
from qadence2_platforms.backends.fresnel1.sequence import Fresnel1
from qadence2_platforms.backends.pyqtorch.interface import Interface as PyQInterface
from qadence2_platforms.backends.pyqtorch.functions import (
    PyQObservablesParser,
    parse_native_observables as pyq_parse_nat_obs,
)
from qadence2_platforms.backends.utils import InputType


# CONSTANTS
N_SHOTS = 3_000
ATOL = 0.06 * N_SHOTS
SMALL_ATOL = 0.005 * N_SHOTS
BIG_ATOL = 0.15 * N_SHOTS

TRAP_COORDINATES = [(0.0, 0.0), (0.0, 5.0), (5.0, 0.0), (5.0, 5.0)]
WEIGHTS = [1.0, 0.5, 0.5, 0]

qi = qutip.qeye(2)
qz = qutip.sigmaz()


##################
# PYQTORCH TESTS #
##################


@pytest.mark.parametrize(
    "interface, n_qubits, expr_obs, pyq_obs, wrong_pyq_obs",
    [
        (
            "pyq_interface1",
            2,
            Z(0),
            pyq.Observable([PZ(0)]),
            pyq.Observable([PZ(1)]),
        ),
        (
            "pyq_interface1",
            2,
            Z(0).__kron__(Z(1)),
            pyq.Observable(pyq.Sequence([PZ(0), PZ(1)])),
            pyq.Observable(pyq.Sequence([PZ(1), PZ(2)])),
        ),
        (
            "pyq_interface1",
            2,
            Z(0).__kron__(Z(1)),
            pyq.Observable(pyq.Sequence([PZ(1), PZ(0)])),
            pyq.Observable(pyq.Sequence([PZ(0), PZ(3)])),
        ),
        (
            "pyq_interface1",
            2,
            Z(0) + Z(1),
            pyq.Observable([pyq.Add([PZ(0), PZ(1)])]),
            pyq.Observable([pyq.Add([PZ(1), PZ(1)])]),
        ),
    ],
)
def test_pyq_observables(
    interface: PyQInterface,
    n_qubits: int,
    expr_obs: InputType,
    pyq_obs: PyQObservable,
    wrong_pyq_obs: PyQObservable,
    request: Any,
) -> None:
    parsed_obs = PyQObservablesParser.build(observable=expr_obs)
    assert parsed_obs.qubit_support == pyq_obs.qubit_support
    assert hash(parsed_obs) == hash(pyq_obs)
    assert hash(parsed_obs) == hash(pyq_parse_nat_obs(expr_obs))
    assert not (hash(parsed_obs) == hash(wrong_pyq_obs))


def test_pyq_obs_parsing() -> None:
    assert hash(PyQObservablesParser._add_op(Z(0) + Z(1))) == hash(pyq.Add([PZ(0), PZ(1)]))
    assert hash(PyQObservablesParser._mul_op(Z(0) * Z(1))) == hash(pyq.Sequence([PZ(0), PZ(1)]))
    assert hash(PyQObservablesParser._kron_op(Z(0).__kron__(Z(1)))) == hash(
        pyq.Sequence([PZ(0), PZ(1)])
    )
    assert hash(PyQObservablesParser._get_op(Z(0))) == hash(pyq.Z(0))
    assert hash(tuple(PyQObservablesParser._iterate_over_obs([Z(0), Z(1)]))) == hash(
        (pyq.Z(0), pyq.Z(1))
    )


##################
# FRESNEL1 TESTS #
##################


def test_dyn_pulse(fresnel1_sequence1: PulserSequence, fresnel1_register1: RegisterLayout) -> None:
    seq = PulserSequence(fresnel1_register1, Fresnel1)  # type: ignore
    seq.declare_channel("global", "rydberg_global")
    dyn_pulse(seq, 1.0, 1.0, 0.0, 0.0)

    # max_amp is about 12.56 rad/µs
    max_amp = seq.device.channels["rydberg_global"].max_amp

    # total_duration is 500 from dyn_pulse duration with
    # duration unit 1.0 + eom pulse + turning eom on and off
    total_duration = 740
    assert seq.get_duration() == total_duration

    with pytest.raises(Exception):
        dyn_pulse(seq, 1000.0, 1.0, 0.0, 0.0)

    with pytest.raises(Exception):
        dyn_pulse(seq, 1.0, 1000.0, 0.0, 0.0)

    interface = Fresnel1Interface(seq, set())
    res = interface.sample(shots=N_SHOTS)
    assert np.allclose(res["01"], res["10"], atol=ATOL)


def test_dyn_wait(fresnel1_sequence1: PulserSequence, fresnel1_register1: RegisterLayout) -> None:
    seq = PulserSequence(fresnel1_register1, Fresnel1)  # type: ignore
    seq.declare_channel("global", "rydberg_global")
    dyn_wait(seq, 1.0)

    interface = Fresnel1Interface(seq, set())
    res = interface.sample(shots=N_SHOTS)
    assert np.allclose(res["01"], res["10"], atol=ATOL)


@pytest.mark.parametrize("direction", [Direction.X, Direction.Y, 0.0])
@pytest.mark.parametrize("angle", [np.pi, np.pi / 2])
def test_rotation(direction: Direction, angle: float) -> None:
    layout = AnalogDevice.calibrated_register_layouts["TriangularLatticeLayout(61, 5.0µm)"]
    register = layout.define_register(0)
    seq = PulserSequence(register, Fresnel1)  # type: ignore
    seq.declare_channel("global", "rydberg_global")
    rotation(seq, np.pi, Direction.X)

    interface1 = Fresnel1Interface(seq, set())
    res = interface1.sample(shots=N_SHOTS)
    assert np.allclose(res["1"], N_SHOTS, atol=SMALL_ATOL)

    x(seq)
    interface2 = Fresnel1Interface(seq, set())
    res = interface2.sample(shots=N_SHOTS)
    assert np.allclose(res["0"], N_SHOTS, atol=SMALL_ATOL)


def test_apply_local_shifts(model1: Model, fresnel1_sequence1: PulserSequence) -> None:
    register_layout = RegisterLayout(TRAP_COORDINATES)  # type: ignore
    detuning_map = register_layout.define_detuning_map({i: WEIGHTS[i] for i in range(4)})

    register = Register.from_coordinates(TRAP_COORDINATES, center=False, prefix="q")
    seq = PulserSequence(register, Fresnel1)  # type: ignore
    seq.declare_channel("global", "rydberg_global")
    seq.config_detuning_map(detuning_map, "dmm_0")

    x(seq)
    apply_local_shifts(seq)
    interface2 = Fresnel1Interface(seq, set())
    res = interface2.sample(shots=N_SHOTS)
    assert np.allclose(res["1001"], res["0110"], atol=ATOL)


def test_local_pulse(fresnel1_sequence1: PulserSequence) -> None:
    register_layout = RegisterLayout(TRAP_COORDINATES)  # type: ignore
    detuning_map = register_layout.define_detuning_map({i: WEIGHTS[i] for i in range(4)})

    register = Register.from_coordinates(TRAP_COORDINATES, center=False, prefix="q")
    seq = PulserSequence(register, Fresnel1)  # type: ignore
    seq.declare_channel("global", "rydberg_global")
    seq.config_detuning_map(detuning_map, "dmm_0")

    x(seq)
    local_pulse(seq, 1.0, 1.0)
    interface2 = Fresnel1Interface(seq, set())
    res = interface2.sample(shots=N_SHOTS)
    assert np.allclose(res["1001"], res["0110"], atol=ATOL)


def test_local_pulse_core(fresnel1_sequence1: PulserSequence) -> None:
    register_layout = RegisterLayout(TRAP_COORDINATES)  # type: ignore
    detuning_map = register_layout.define_detuning_map({i: WEIGHTS[i] for i in range(4)})

    register = Register.from_coordinates(TRAP_COORDINATES, center=False, prefix="q")
    seq = PulserSequence(register, Fresnel1)  # type: ignore
    seq.declare_channel("global", "rydberg_global")
    seq.config_detuning_map(detuning_map, "dmm_0")

    max_amp = seq.device.channels["rydberg_global"].max_amp
    time_scale = 1000 * 2 * np.pi / max_amp

    local_pulse_core(seq, 1.0, time_scale, 1.0, False)
    x(seq)
    local_pulse_core(seq, Duration.FILL, time_scale, 0.5, True)

    with pytest.raises(Exception):
        local_pulse_core(seq, Duration.FILL, time_scale, 0.5, False)

    with pytest.raises(ValueError):
        local_pulse_core(seq, Duration.FILL, 1.0, 0.5, True)

    with pytest.raises(ValueError):
        local_pulse_core(seq, 1.0, 1.0, 1.0)

    interface2 = Fresnel1Interface(seq, set())
    res = interface2.sample(shots=N_SHOTS)
    assert np.allclose(res["0001"], N_SHOTS, atol=BIG_ATOL)


def test_h(fresnel1_sequence1: PulserSequence) -> None:
    layout = AnalogDevice.calibrated_register_layouts["TriangularLatticeLayout(61, 5.0µm)"]
    register = layout.define_register(0)
    seq = PulserSequence(register, Fresnel1)  # type: ignore
    seq.declare_channel("global", "rydberg_global")
    h(seq, np.pi)

    with pytest.raises(Exception):
        h(seq, 1000.0)

    with pytest.raises(Exception):
        h(seq, -1.0)

    with pytest.raises(Exception):
        h(seq, 1.0, "GLOBAL")

    interface1 = Fresnel1Interface(seq, set())
    res = interface1.sample(shots=N_SHOTS)
    print(res)
    assert np.allclose(res["1"], res["0"], atol=BIG_ATOL)


def test_rx(fresnel1_sequence1: PulserSequence) -> None:
    layout = AnalogDevice.calibrated_register_layouts["TriangularLatticeLayout(61, 5.0µm)"]
    register = layout.define_register(0)
    seq = PulserSequence(register, Fresnel1)  # type: ignore
    seq.declare_channel("global", "rydberg_global")
    rx(seq, np.pi)

    with pytest.raises(Exception):
        rx(seq, Direction.X)

    with pytest.raises(Exception):
        rx(seq, "theta")

    interface1 = Fresnel1Interface(seq, set())
    res = interface1.sample(shots=N_SHOTS)
    print(res)
    assert np.allclose(res["1"], N_SHOTS, atol=SMALL_ATOL)


def test_ry(fresnel1_sequence1: PulserSequence) -> None:
    layout = AnalogDevice.calibrated_register_layouts["TriangularLatticeLayout(61, 5.0µm)"]
    register = layout.define_register(0)
    seq = PulserSequence(register, Fresnel1)  # type: ignore
    seq.declare_channel("global", "rydberg_global")
    ry(seq, np.pi)

    with pytest.raises(Exception):
        ry(seq, Direction.X)

    with pytest.raises(Exception):
        ry(seq, "theta")

    interface1 = Fresnel1Interface(seq, set())
    res = interface1.sample(shots=N_SHOTS)
    print(res)
    assert np.allclose(res["1"], N_SHOTS, atol=SMALL_ATOL)


@pytest.mark.parametrize(
    "interface, n_qubits, expr_obs, qutip_obs",
    [
        ("fresnel1_interface", 2, Z(0), qtensor(qz, qi)),
        ("fresnel1_interface", 2, Z(0).__kron__(Z(1)), qtensor(qz, qz)),
        ("fresnel1_interface", 2, Z(0) + Z(1), qtensor(qz, qi) + qtensor(qi, qz)),
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
