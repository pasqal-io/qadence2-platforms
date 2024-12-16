from __future__ import annotations

from enum import Enum, auto
from functools import reduce
from typing import Iterable, cast

import numpy as np
import qutip

from qadence2_platforms.backends.utils import InputType, Support

DEFAULT_AMPLITUDE = 4 * np.pi
DEFAULT_DETUNING = 10 * np.pi

# TODO: re-introduce `Support` to account for "local" and "global" on the `channel` arg


class Direction(Enum):
    X = auto()
    Y = auto()


class Duration(Enum):
    FILL = auto()


###############
# OBSERVABLES #
###############


def base_parse_native_observables(
    num_qubits: int, observable: list[InputType] | InputType
) -> list[qutip.Qobj]:
    """
    Function to be called by `Interface`'s `expectation` method on Pulser-based backends
    using QuTiP emulator.

    Args:
        num_qubits (int): number of qubits
        observable (list[InputType] | InputType): the input expression. Any
            qadence2-expressions expression compatible object, with the same
            methods, or a list of it

    Returns:
        A QuTiP object with the Hilbert space compatible with `num_qubits`
    """
    return BaseQuTiPObservablesParser.build(num_qubits, observable)


class BaseQuTiPObservablesParser:
    """
    Convert InputType object to Qutip native quantum objects for simulation on QuTiP.

    It is intended to be used on the expectation method of Pulser-based interface classes.
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
        """
        Convert an expression into a native QuTiP object. A simple symbol,
        a quantum operator, and an operation (addition, multiplication or
        kron tensor) are valid objects.

        Args:
            num_qubits (int): the number of qubits to create the qutip object to
            op (InputType): the input expression. Any qadence2-expressions expression
                compatible object, with the same methods

        Returns:
            A QuTiP object with the Hilbert space compatible with `num_qubits`
        """

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
        """
        Parses an input expression or list of expressions into a native QuTiP object.

        Args:
            num_qubits (int): the number of qubits to create the qutip object to
            observables (list[InputType], InputType): the input expression. Any
                qadence2-expressions expression compatible object, with the same
                methods

        Returns:
            A QuTiP object with the Hilbert space compatible with `num_qubits`
        """
        if not isinstance(observables, list):
            return [cls._get_op(num_qubits, observables)]
        return cls._iterate_over_obs(num_qubits, observables)
