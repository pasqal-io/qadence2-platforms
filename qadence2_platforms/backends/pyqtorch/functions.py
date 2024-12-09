from __future__ import annotations

from typing import Iterable, cast

from pyqtorch.primitives import Primitive
from torch.nn import Module
import pyqtorch
from pyqtorch import Sequence, Add
from pyqtorch.hamiltonians import Observable

from qadence2_platforms.backends.utils import InputType, Support


def parse_native_observables(observable: list[InputType] | InputType) -> Observable:
    return PyQObservablesParser.build(observable)


class PyQObservablesParser:
    """
    Convert InputType object observables into native PyQTorch object, especially for
    running `expectation` method from PyQTorch interface class. InputType can be
    qadence2-expressions or any other module that implement the same methods.
    """

    @classmethod
    def _add_op(cls, op: InputType) -> Module:
        return Add([cls._get_op(cast(InputType, k)) for k in cast(Iterable, op.args)])

    @classmethod
    def _mul_op(cls, op: InputType) -> Module:
        return Sequence([cls._get_op(cast(InputType, k)) for k in cast(Iterable, op.args)])

    @classmethod
    def _kron_op(cls, op: InputType) -> Module:
        return cls._mul_op(op)

    @classmethod
    def _get_op(cls, op: InputType) -> Primitive | Module:
        """
        Convert an expression into a native PyQTorch object. A simple symbol,
        a quantum operator, and an operation (addition, multiplication or
        kron tensor) are valid objects.

        Args:
            op (InputType): the input expression. Any qadence2-expressions
                expression compatible object or object with same methods.

        Returns:
            A Primitive or torch.nn.Module object.
        """

        if op.is_symbol is True:
            symbol: str = cast(str, op.args[0])
            return getattr(pyqtorch, symbol.upper(), None)

        if op.is_quantum_operator is True:
            primitive: Primitive = cast(Primitive, cls._get_op(cast(InputType, op.args[0])))
            support: Support = cast(Support, op.args[1])

            if support:
                target: int = support.target[0]
                control: list[int] = support.control

                if control:
                    native_op = primitive(target=target, control=control)
                else:
                    native_op = primitive(target=target)

                return native_op

        if op.is_addition is True:
            return cls._add_op(op)

        if op.is_multiplication is True or op.is_kronecker_product is True:
            return cls._mul_op(op)

        raise NotImplementedError(
            f"could not retrieve the expression {op} ({type(op)}) from the observables"
        )

    @classmethod
    def _iterate_over_obs(cls, op: list[InputType] | InputType) -> list[Module]:
        if isinstance(op, Iterable):
            return [cls._get_op(arg) for arg in op]

        args: Iterable = cast(Iterable, op.args)
        return [cls._get_op(arg) for arg in args]

    @classmethod
    def build(cls, observable: list[InputType] | InputType) -> Observable:
        """
        Parses an input expression or list of expressions into a native PyQTorch object.

        Args:
            observable (list[InputType] | InputType): the input expression. Any
                qadence2-expressions expression compatible object or object with same
                methods.

        Returns:
            An PyQTorch Observable object.
        """

        res: list[Module] | Module
        if isinstance(observable, list):
            res = cls._iterate_over_obs(observable)
        else:
            res = [cls._get_op(observable)]
        return Observable(res)
