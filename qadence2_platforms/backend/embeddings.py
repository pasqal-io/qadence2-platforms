from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Any

from qadence2_platforms import Model
from qadence2_platforms.types import (
    ParameterResultType,
    DType,
    EmbeddingMappingResultType
)


class ParameterBuffer(ABC, Generic[DType, ParameterResultType]):
    """
    A generic parameter class to hold all root parameters passed by the user or
    trainable variational parameters.
    """

    @property
    @abstractmethod
    def dtype(self) -> DType:
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> ParameterResultType:
        raise NotImplementedError()


class EmbeddingModule(ABC, Generic[EmbeddingMappingResultType]):
    """
    A generic module class to hold and handle the parameters and expressions
    functions coming from the `Model`. It may contain the list of user input
    parameters, as well as the trainable variational parameters and the
    evaluated functions from the data types being used, i.e. torch, numpy, etc.
    """

    @abstractmethod
    def __init__(self, model: Model):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def name_mapping(self) -> EmbeddingMappingResultType:
        raise NotImplementedError()
