from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Literal, TypeVar

SequenceType = TypeVar("SequenceType")
ResultType = TypeVar("ResultType")


class AbstractInterface(ABC, Generic[SequenceType, ResultType]):

    @property
    @abstractmethod
    def info(self) -> dict[str, Any]:
        pass

    @property
    @abstractmethod
    def sequence(self) -> SequenceType:
        pass

    @abstractmethod
    def set_parameters(self, params: dict[str, float]) -> None:
        pass

    @abstractmethod
    def add_noise(self, model: Literal["SPAM"]) -> None:
        pass

    @abstractmethod
    def run(self, shots: int | None = None, callback: Callable | None = None, **kwargs: Any) -> ResultType:
        pass
