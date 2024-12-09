from __future__ import annotations

from typing import Any
import pytest

from qadence2_platforms.backends.utils import Support, InputType
from qadence2_expressions.core.support import Support as ExprSupport
from qadence2_expressions import parameter, X, Z, RX
from qadence2_ir.types import Support as IRSupport


@pytest.mark.parametrize(
    "data",
    [
        ExprSupport(target=(1,)),
        ExprSupport(target=(0,), control=(3,)),
        ExprSupport(target=(2,), control=(5, 6)),
    ],
)
def test_support(data: Support) -> None:
    assert isinstance(data, Support)


@pytest.mark.parametrize(
    "data",
    [
        (0,),
        (0, 1, 2),
        ((0,),),
        ((0,), (1,)),
        ((0,), (1, 2)),
        IRSupport(target=(0,)),
        IRSupport(target=(0,), control=(1,)),
    ],
)
def test_not_support(data: Any) -> None:
    assert not isinstance(data, Support)


@pytest.mark.parametrize(
    "data",
    [
        X(0),
        X(0) * X(1),
        X(1) * Z(1),
        X(0) * Z(1),
        RX(parameter("a"))(0),
        parameter("a") + parameter("b"),
        parameter("a") * X(0),
    ],
)
def test_inputtype(data: InputType) -> None:
    assert isinstance(data, InputType)
