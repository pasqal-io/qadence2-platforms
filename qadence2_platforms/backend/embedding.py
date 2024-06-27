from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Any, Optional

from qadence2_platforms import Model
from qadence2_platforms.types import (
    ParameterType,
    DType,
    EmbeddingType,
    EmbeddingMappingResultType,
)


class ParameterBufferApi(ABC, Generic[DType, ParameterType]):
    """
    A generic parameter class to hold all root parameters passed by the user or
    trainable variational parameters.
    """
    _dtype: DType
    vparams: dict[str, ParameterType]
    fparams: dict[str, Optional[ParameterType]]

    @property
    def dtype(self) -> DType:
        return self._dtype


class EmbeddingModuleApi(ABC, Generic[EmbeddingType, EmbeddingMappingResultType]):
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
    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, EmbeddingType]:
        raise NotImplementedError()

    @abstractmethod
    def name_mapping(self) -> EmbeddingMappingResultType:
        raise NotImplementedError()
