# type: ignore
# TODO: 1. remove the line above after implementing the backend
# TODO: 2. remove these comments

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TypeVar, Iterable

from qadence2_platforms.abstracts import (
    AbstractInterface,
)

logger = logging.getLogger(__name__)


# TODO: replace the following types by relevant types for this custom interface
# TODO: erase these TODO comments
ArrayType = TypeVar("ArrayType")
SequenceType = TypeVar("SequenceType")
ParameterType = TypeVar("ParameterType")
RunResultType = TypeVar("RunResultType")
SampleResultType = TypeVar("SampleResultType")
ExpectationResultType = TypeVar("ExpectationResultType")


# TODO: replace the new types in the respective order in the `AbstractInterface` below:
# TODO: erase these TODO comments
class Interface(
    AbstractInterface[
        ArrayType,
        SequenceType,
        ParameterType,
        RunResultType,
        SampleResultType,
        ExpectationResultType,
    ],
):
    """
    TODO: 1.

    fill the methods with relevant arguments and code logic

    Hint: Check the built-in backends for reference on how to write this class.

    TODO: 2. replace this docstring with a description of this custom interface class
    """

    def __init__(self) -> None:
        # TODO: 1. implement this method logic
        # TODO: 2. include relevant arguments on the method
        # TODO: 3. write docstring if needed
        # TODO: 4. erase these TODO comments
        pass

    @property
    def info(self) -> dict[str, Any]:
        # TODO: 1. implement this method logic
        # TODO: 2. erase these TODO comments
        pass

    @property
    def sequence(self) -> SequenceType:
        # TODO: 1. implement this method logic
        # TODO: 2. erase these TODO comments
        pass

    def parameters(self) -> Iterable[Any]:
        # TODO: 1. implement this method logic
        # TODO: 2. erase these TODO comments
        pass

    def set_parameters(self, params: dict[str, ParameterType]) -> None:
        # TODO: 1. implement this method logic
        # TODO: 2. include relevant arguments on the method
        # TODO: 3. do not remove the existing arguments!
        # TODO: 4. replace return generic type for relevant type
        # TODO: 5. write docstring if needed
        # TODO: 6. erase these TODO comments
        pass

    def run(
        self,
        *,
        values: dict[str, ArrayType] | None = None,
        callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> RunResultType:
        # TODO: 1. implement this method logic
        # TODO: 2. include relevant arguments on the method
        # TODO: 3. do not remove the existing arguments!
        # TODO: 4. replace return generic type for relevant type
        # TODO: 5. write docstring!
        # TODO: 6. erase these TODO comments
        pass

    def sample(
        self,
        *,
        values: dict[str, ArrayType] | None = None,
        shots: int | None = None,
        callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> SampleResultType:
        # TODO: 1. implement this method logic
        # TODO: 2. include relevant arguments on the method
        # TODO: 3. do not remove the existing arguments!
        # TODO: 4. replace return generic type for relevant type
        # TODO: 5. write docstring!
        # TODO: 6. erase these TODO comments
        pass

    def expectation(
        self,
        *,
        values: dict[str, ArrayType] | None = None,
        observable: Any | None = None,
        callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> ExpectationResultType:
        # TODO: 1. implement this method logic
        # TODO: 2. include relevant arguments on the method
        # TODO: 3. do not remove the existing arguments!
        # TODO: 4. replace return generic type for relevant type
        # TODO: 5. write docstring!
        # TODO: 6. erase these TODO comments
        pass
