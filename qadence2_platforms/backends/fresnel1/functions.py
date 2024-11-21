from __future__ import annotations

from enum import Enum, auto
from functools import reduce
from typing import Any, Iterable, cast

import numpy as np
import qutip
from pulser.parametrized.variable import VariableItem
from pulser.sequence import Sequence
from pulser.waveforms import ConstantWaveform

from qadence2_platforms.backends.utils import InputType, Support

DEFAULT_AMPLITUDE = 4 * np.pi
DEFAULT_DETUNING = 10 * np.pi

# TODO: re-introduce `Support` to account for "local" and "global" on the `channel` arg


class Direction(Enum):
    X = auto()
    Y = auto()


class Duration(Enum):
    FILL = auto()


# pulse function mapping.
# keys are the `qadence2_ir.Model` standard, values are the function names below
PULSE_FN_MAP = {
    "rx": "rx",
    "ry": "ry",
    "not": "x",
    "h": "h",
}


def dyn_pulse(
    sequence: Sequence,
    duration: VariableItem | float,
    amplitude: VariableItem | float,
    detuning: VariableItem | float,
    phase: VariableItem | float,
    **_: Any,
) -> None:
    """
    Dynamic pulse to simulate a specific time-dependent hamiltonian for neutral-atom devices.

    :param sequence: a `pulser.sequence.Sequence` instance
    :param duration: duration of the pulse in nanoseconds
    :param amplitude: amplitude of the pulse in rad/µs
    :param detuning: detuning of the pulse in rad/s
    :param phase: phase in rad
    """
    max_amp = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    max_abs_detuning = (
        sequence.device.channels["rydberg_global"].max_abs_detuning or DEFAULT_DETUNING
    )

    duration *= 1000 * 2 * np.pi / max_amp  # type: ignore
    amplitude *= max_amp  # type: ignore
    detuning *= max_abs_detuning  # type: ignore

    sequence.enable_eom_mode("global", amp_on=amplitude, detuning_on=detuning)
    sequence.add_eom_pulse("global", duration=duration, phase=phase)  # type: ignore
    sequence.disable_eom_mode("global")


def rx(
    sequence: Sequence,
    angle: VariableItem | float,
    **_: Any,
) -> None:
    rotation(sequence, angle, Direction.X)


def ry(
    sequence: Sequence,
    angle: VariableItem | float,
    **_: Any,
) -> None:
    rotation(sequence, angle, Direction.Y)


def x(sequence: Sequence, **_: Any) -> None:
    rotation(sequence, angle=np.pi, direction=Direction.X)


def h(
    sequence: Sequence,
    duration: VariableItem | float = 1.0,
    support_list: str = "global",
    **_: Any,
) -> None:
    amplitude = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    duration *= 1000 * 2 * np.pi / amplitude
    detuning = np.pi

    sequence.enable_eom_mode(
        "global",
        amp_on=amplitude,
        correct_phase_drift=True,
        detuning_on=detuning
    )
    sequence.add_eom_pulse(
        "global", duration=int(duration), phase=np.pi / 2, post_phase_shift=np.pi
    )
    sequence.disable_eom_mode(support_list)


def rotation(
    sequence: Sequence,
    angle: VariableItem | float,
    direction: Direction | float,
    **_: Any,
) -> None:

    amplitude = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    duration = 1000 * angle / amplitude
    detuning = 0

    phase: Any
    match direction:
        case Direction.X:
            phase = 0
        case Direction.Y:
            phase = np.pi / 2
        case _:
            phase = direction

    sequence.enable_eom_mode("global", amp_on=amplitude, detuning_on=detuning)
    sequence.add_eom_pulse("global", duration=duration, phase=phase)  # type: ignore
    sequence.disable_eom_mode("global")


def dyn_wait(
    sequence: Sequence,
    duration: VariableItem | float,
    *args: Any,
    **kwargs: Any,
) -> None:
    max_amp = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE

    duration *= 1000 * 2 * np.pi / max_amp  # type: ignore

    sequence.delay(int(duration), "global")  # type: ignore


def apply_local_shifts(sequence: Sequence, **_: Any) -> None:
    max_abs_detuning = (
        sequence.device.channels["rydberg_global"].max_abs_detuning or DEFAULT_DETUNING
    )
    time_scale = 1000 * 2 * np.pi / max_abs_detuning
    local_pulse_core(sequence, duration=1.0, time_scale=time_scale, detuning=1.0, concurrent=False)


def local_pulse(
    sequence: Sequence,
    duration: VariableItem | float | Duration,
    detuning: VariableItem | float,
    concurrent: bool = False,
    **_: Any,
) -> None:
    max_amp = sequence.device.channels["rydberg_global"].max_amp or DEFAULT_AMPLITUDE
    time_scale = 1000 * 2 * np.pi / max_amp
    local_pulse_core(sequence, duration, time_scale, detuning, concurrent)


def local_pulse_core(
    sequence: Sequence,
    duration: VariableItem | float | Duration,
    time_scale: float,
    detuning: VariableItem | float,
    concurrent: bool = False,
    **kwargs: Any,
) -> None:
    max_abs_detuning = (
        sequence.device.channels["rydberg_global"].max_abs_detuning or DEFAULT_DETUNING
    )

    if duration == Duration.FILL:
        if not concurrent:
            raise SyntaxError("The option `fill` can only be used on the `concurrent` mode")

        duration = sequence.get_duration("global") - sequence.get_duration("dmm_0")
    else:
        duration *= time_scale  # type: ignore

    detuning *= -1 * max_abs_detuning  # type: ignore

    sequence.add_dmm_detuning(
        ConstantWaveform(duration, detuning),
        "dmm_0",
        "no-delay" if concurrent else "min-delay",
    )


def parse_native_observables(
    num_qubits: int, observable: list[InputType] | InputType
) -> list[qutip.Qobj]:
    return QuTiPObservablesParser.build(num_qubits, observable)


class QuTiPObservablesParser:
    """
    Convert InputType object to Qutip native quantum objects for simulation on QuTiP.

    It is intended to be used on expectation method of the Fresnel1 interface class.
    InputType can be qadence2-expressions expression or any other module with the same
    methods.
    """

    operators_mapping = {
        "I": qutip.qeye(2),
        "Z": qutip.sigmaz(),
    }

    @classmethod
    def _compl_tensor_op(cls, num_qubits: int, expr: InputType) -> qutip.Qobj:
        """
        Use it for pure kron operations or for single input operator that needs to match
        a bigger Hilbert space, e.g. `expr = Z(0)` but the number of qubits is 3.

        Args:
            num_qubits (int): the number of qubits to create the qutip object to
            expr (InputType): the input expression. Any qadence2-expressions expression
                compatible object, with the same methods

        Returns:
            A QuTiP object with the Hilbert space compatible with `num_qubits`
        """

        op: qutip.Qobj
        arg: InputType

        if expr.subspace:
            native_ops: list[qutip.Qobj] = []
            support_set: set = expr.subspace.subspace

            for k in range(num_qubits):
                if k in support_set:
                    sub_num_qubits: int = cast(Support, expr.args[1]).max_index
                    arg = cast(InputType, expr.args[0])
                    op = cls._get_op(sub_num_qubits, arg)

                else:
                    op = cls.operators_mapping["I"]

                native_ops.append(op)

            return qutip.tensor(*native_ops)

        arg = cast(InputType, expr.args[0])
        op = cls._get_op(num_qubits, arg)
        return qutip.tensor(*([op] * num_qubits))

    @classmethod
    def _arith_tensor_op(cls, num_qubits: int, expr: InputType) -> qutip.Qobj:
        """
        Use it for the arithmetic operations addition and multiplication that need to
        have an extended Hilbert space compatible with `num_qubits`.

        Args:
            num_qubits (int): the number of qubits to create the qutip object to
            expr (InputType): the input expression. Any qadence2-expressions expression
                compatible object, with the same methods

        Returns:
            A QuTiP object with the Hilbert space compatible with `num_qubits`
        """

        subspace: set = cast(Support, expr.subspace).subspace
        super_space: set = set(range(num_qubits))

        if super_space.issuperset(subspace):
            native_ops: list[qutip.Qobj] = []
            arg_subspace: set = cast(Support, expr.args[1]).subspace

            for k in range(num_qubits):
                if k in arg_subspace:
                    sub_num_qubits: int = cast(int, expr.args[1])
                    arg: InputType = cast(InputType, expr.args[0])
                    op: qutip.Qobj = cls._get_op(sub_num_qubits, arg)
                    native_ops.append(op)
                else:
                    native_ops.append(cls.operators_mapping["I"])

            return qutip.tensor(*native_ops)

        raise ValueError(
            f"subspace of the object ({subspace}) is bigger than the "
            f"sequence space ({super_space})"
        )

    @classmethod
    def _get_op(cls, num_qubits: int, op: InputType) -> qutip.Qobj | None:
        if op.is_symbol is True:
            symbol: str = cast(str, op.args[0])
            return cls.operators_mapping[symbol]

        op_arg: qutip.Qobj

        if op.is_quantum_operator is True:
            sub_num_qubits: int = cast(Support, op.args[1]).max_index + 1

            if sub_num_qubits < num_qubits:
                op_arg = cls._compl_tensor_op(num_qubits, op)

            else:
                arg: InputType = cast(InputType, op.args[0])
                op_arg = cls._get_op(num_qubits, arg)

            return op_arg

        ops: list[qutip.Qobj] = []
        args: Iterable = cast(Iterable, op.args)

        if op.is_addition is True:

            for arg in args:
                ops.append(cls._arith_tensor_op(num_qubits, arg))

            return reduce(lambda a, b: a + b, ops)

        if op.is_multiplication is True:

            for arg in args:
                ops.append(cls._arith_tensor_op(num_qubits, arg))

            return reduce(lambda a, b: a * b, ops)

        if op.is_kronecker_product is True:
            return cls._compl_tensor_op(num_qubits, op)

        raise NotImplementedError(
            f"could not retrieve the expression {op} ({type(op)}) from the observables"
        )

    @classmethod
    def _iterate_over_obs(cls, n_qubits: int, op: Iterable | InputType) -> list[qutip.Qobj]:
        if isinstance(op, Iterable):
            return [cls._get_op(n_qubits, arg) for arg in op]

        args: Iterable = cast(Iterable, op.args)
        return [cls._get_op(n_qubits, cast(InputType, arg)) for arg in args]

    @classmethod
    def build(cls, num_qubits: int, observables: list[InputType] | InputType) -> list[qutip.Qobj]:
        if not isinstance(observables, list):
            return [cls._get_op(num_qubits, observables)]

        raise NotImplementedError("list of observables to be implemented")
