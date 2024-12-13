from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Generic, Iterable, TypeVar

ArrayType = TypeVar("ArrayType")
SequenceType = TypeVar("SequenceType")
ParameterType = TypeVar("ParameterType")
RunResultType = TypeVar("RunResultType")
SampleResultType = TypeVar("SampleResultType")
ExpectationResultType = TypeVar("ExpectationResultType")


class RunEnum(Enum):
    """
    Enum class to be used whenever an Interface class method need to specify
    how to execute the expression: through `run`, `sample`, or `expectation`.
    """

    RUN = auto()
    SAMPLE = auto()
    EXPECTATION = auto()


class OnEnum(Enum):
    """
    Enum class to be used whenever an Interface class method (such as `run`) needs to
    specify where to run the code: on emulator or qpu.
    """

    EMULATOR = auto()
    QPU = auto()


class AbstractInterface(
    ABC,
    Generic[
        ArrayType,
        SequenceType,
        ParameterType,
        RunResultType,
        SampleResultType,
        ExpectationResultType,
    ],
):
    """
    An abstract base class that defines interface essential methods.

    It should be inherited by any class that needs to implement a backend, for instance
    `pyqtorch` and `fresnel1` (`pulser` using `qutip` emulator). It is not only used by
    the package itself, but users who want to implement or test new backends should
    also make use of it.
    """

    @property
    @abstractmethod
    def info(self) -> dict[str, Any]:
        """
        Gives any relevant information about the interface data, such as `device`,
        `register`, etc.

        :return: dictionary with the relevant information.
        """
        pass

    @property
    @abstractmethod
    def sequence(self) -> SequenceType:
        """
        Outputs the backends-native sequence.

        :return: The defined backends-native sequence.
        """
        pass

    @abstractmethod
    def parameters(self) -> Iterable[Any]:
        """
        Get the parameters from the backend as an iterable.

        Returns:
            Parameters as an iterable.
        """
        pass

    @abstractmethod
    def set_parameters(self, params: dict[str, ParameterType]) -> None:
        """
        Sets valid parameters for the backends to use it during simulation/execution step.

        :param params: the fixed parameters to be used by the backends.
        """
        pass

    @abstractmethod
    def draw(self, values: dict[str, Any]) -> None:
        """
        May draw the current sequence with the given values.

        Args:
            values (dict[str, Any]): the values to be drawn
        """
        pass

    @abstractmethod
    def run(
        self,
        values: dict[str, ArrayType] | None = None,
        **kwargs: Any,
    ) -> RunResultType:
        """
        Gets the results from the expression computation given the parameters (values),
        and extra arguments.

        :param values: dictionary of user-input parameters
        :param kwargs: any extra argument that are backends specific can be included in the
            child method.
        :return: any result type according to what is expected by the backends `run` method
        """
        pass

    @abstractmethod
    def sample(
        self,
        values: dict[str, ArrayType] | None = None,
        shots: int | None = None,
        **kwargs: Any,
    ) -> SampleResultType:
        """
        Samples the computed result given the expression, the parameters (values), number of
        shots, and extra arguments.

        :param values: dictionary of user-input parameters
        :param shots: number of shots
        :param kwargs: any extra argument that are backends specific can be included in the
            child method
        :return: any result type according to what is expected by the backends `sample` method
        """
        pass

    @abstractmethod
    def expectation(
        self,
        values: dict[str, ArrayType] | None = None,
        observable: Any | None = None,
        **kwargs: Any,
    ) -> ExpectationResultType:
        """
        Computes the expectation value for observable(s) given the parameters (values),
        and extra arguments.

        :param values: dictionary of user-input parameters
        :param observable: list of observables
        :param kwargs: any extra argument that are backends specific can be included in the
            child method
        :return: any result type according to what is expected by the backends `expectation` method
        """
        pass
