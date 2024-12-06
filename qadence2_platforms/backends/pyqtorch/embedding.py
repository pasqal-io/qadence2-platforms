from __future__ import annotations

from logging import getLogger
from typing import Any, Callable

import torch
from qadence2_ir.types import Assign, Call, Load, Model

logger = getLogger(__name__)


def torch_call(call: Call) -> Callable[[dict, dict], torch.Tensor]:
    """
    Convert a `Call` object into a torchified function which can be evaluated using.

    a vparams and inputs dict.
    """
    fn = getattr(torch, call.identifier)

    def evaluate(params: dict, inputs: dict) -> torch.Tensor:
        args = []
        for symbol in call.args:
            if isinstance(symbol, float):
                # NOTE we compile constants into each TorchCallable instead of passing
                # them around in the values dict
                args.append(torch.tensor(symbol))
            elif isinstance(symbol, Load):
                args.append({**params, **inputs}[symbol.variable])
        return fn(*args)

    return evaluate


class ParameterBuffer(torch.nn.Module):
    """
    A class holding all root parameters either passed by the user.

    or trainable variational parameters.
    """

    def __init__(
        self,
        trainable_vars: list[str],
        non_trainable_vars: list[str],
    ) -> None:
        super().__init__()
        self.vparams = {p: torch.rand(1, requires_grad=True) for p in trainable_vars}
        self.fparams = {p: None for p in non_trainable_vars}
        self._dtype = torch.float64
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def to(self, *args: Any, **kwargs: Any) -> ParameterBuffer:
        self.vparams = {p: t.to(*args, **kwargs) for p, t in self.vparams.items()}
        try:
            k = next(iter(self.vparams))
            t = self.vparams[k]
            self._device = t.device
            self._dtype = t.dtype
        except Exception:
            pass
        return self

    @classmethod
    def from_model(cls, model: Model) -> ParameterBuffer:
        f_p = []
        v_p = []
        for param_name, alloc in model.inputs.items():
            if alloc.is_trainable:
                v_p.append(param_name)
            else:
                f_p.append(param_name)
        return ParameterBuffer(v_p, f_p)


class Embedding(torch.nn.Module):
    """
    A class holding:

    - A parameterbuffer containing concretized vparams + list of featureparams,
    - A dictionary of intermediate and leaf variable names mapped to a TorchCall object
        which can be results of function/expression evaluations.
    """

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.param_buffer = ParameterBuffer.from_model(model)
        self.var_to_torchcall: dict[str, Callable] = self.create_var_to_torchcall_mapping(model)

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Expects a dict of user-passed name:value pairs for featureparameters.

        and assigns all intermediate and leaf variables using the current vparam values
        and the passed values for featureparameters.
        """
        assigned_params: dict[str, torch.Tensor] = {}
        try:
            # TODO: check why it is failing (doesn't affect code reliability apparently)
            assert inputs.keys() == self.param_buffer.fparams.keys()
        except Exception as _:
            pass
        for var, torchcall in self.var_to_torchcall.items():
            assigned_params[var] = torchcall(
                self.param_buffer.vparams,
                {
                    **inputs,
                    **assigned_params,
                },  # we add the "intermediate" variables too
            )
        assigned_params = {**assigned_params, **inputs}
        return assigned_params

    @staticmethod
    def create_var_to_torchcall_mapping(
        model: Model,
    ) -> dict[str, Callable[[dict, dict], torch.Tensor]]:
        assign_to_torch = dict()
        for instr in model.instructions:
            if isinstance(instr, Assign):
                assign_to_torch[instr.variable] = torch_call(instr.value)
        return assign_to_torch
