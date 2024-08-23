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
        Outputs the backend-native sequence.

        :return: The defined backend-native sequence.
        """
        pass

    @abstractmethod
    def set_parameters(self, params: dict[str, ParameterType]) -> None:
        """
        Sets valid parameters for the backend to use it during simulation/execution step.

        :param params: the fixed parameters to be used by the backend.
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
        pass
