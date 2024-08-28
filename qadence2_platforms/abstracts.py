from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Optional, TypeVar

ArrayType = TypeVar("ArrayType")
SequenceType = TypeVar("SequenceType")
ParameterType = TypeVar("ParameterType")
RunResultType = TypeVar("RunResultType")
SampleResultType = TypeVar("SampleResultType")
ExpectationResultType = TypeVar("ExpectationResultType")


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
    def set_parameters(self, params: dict[str, ParameterType]) -> None:
        """
        Sets valid parameters for the backends to use it during simulation/execution step.

        :param params: the fixed parameters to be used by the backends.
        """
        pass

    @abstractmethod
    def run(
        self,
        *,
        values: Optional[dict[str, ArrayType]] = None,
        callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> RunResultType:
        """
        Gets the results from the expression computation given the parameters (values),
        callback function (if applicable), and extra arguments.

        :param values: dictionary of user-input parameters
        :param callback: a callback function if necessary to run some extra processing
        :param kwargs: any extra argument that are backends specific can be included in the
            child method.
        :return: any result type according to what is expected by the backends `run` method
        """
        pass

    @abstractmethod
    def sample(
        self,
        *,
        values: Optional[dict[str, ArrayType]] = None,
        shots: Optional[int] = None,
        callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> SampleResultType:
        """
        Samples the computed result given the expression, the parameters (values), number of
        shots, callback function (if applicable), and extra arguments.

        :param values: dictionary of user-input parameters
        :param shots: number of shots
        :param callback: a callback function if necessary to run some extra processing
        :param kwargs: any extra argument that are backends specific can be included in the
            child method
        :return: any result type according to what is expected by the backends `sample` method
        """
        pass

    @abstractmethod
    def expectation(
        self,
        *,
        values: Optional[dict[str, ArrayType]] = None,
        observable: Optional[Any] = None,
        callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> ExpectationResultType:
        """
        Computes the expectation value for observable(s) given the parameters (values),
        callback function (if applicable), and extra arguments.

        :param values: dictionary of user-input parameters
        :param observable: list of observables
        :param callback: a callback function if necessary to run some extra processing
        :param kwargs: any extra argument that are backends specific can be included in the
            child method
        :return: any result type according to what is expected by the backends `expectation` method
        """
        pass
