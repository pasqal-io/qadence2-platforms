from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar

SequenceType = TypeVar("SequenceType")
ParameterType = TypeVar("ParameterType")
ResultType = TypeVar("ResultType")


class AbstractInterface(ABC, Generic[SequenceType, ParameterType, ResultType]):

    @property
    @abstractmethod
    def info(self) -> dict[str, Any]:
        pass

    @property
    @abstractmethod
    def sequence(self) -> SequenceType:
        pass

    @abstractmethod
    def run(
        self,
        *,
        parameters: dict[str, float] | None = None,
        shots: int | None = None,
        callback: Callable | None = None,
        **kwargs: Any,
    ) -> ResultType:
        pass
