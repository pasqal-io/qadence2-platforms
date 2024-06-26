from __future__ import annotations

from logging import getLogger
from typing import Callable

import pyqtorch as pyq
import torch

from qadence2_platforms.qadence_ir import Assign, Call, Load, Model, QuInstruct

logger = getLogger(__name__)
C_OPMap = {
    "add": pyq.Add,
    "mul": pyq.Scale,
    "noncommute": pyq.Sequence,
    "sin": torch.sin,
}
Q_OpMap = {
    "rx": pyq.RX,
    "ry": pyq.RY,
    "rz": pyq.RZ,
    "not": pyq.CNOT,
}


def TorchCallable(call: Call) -> Callable[[dict, dict], torch.Tensor]:
    fn = getattr(torch, call.call)

    def evaluate(params, inputs) -> torch.Tensor:
        args = []
        for symbol in call.args:
            if isinstance(symbol, float):
                args.append(torch.tensor(symbol))
            elif isinstance(symbol, Load):
                args.append({**params, **inputs}[symbol.variable])
        return fn(*args)

    return evaluate


class ParameterBuffer(torch.nn.Module):
    """A class holding all root parameters either passed by the user or trainable variational parameters."""

    def __init__(
        self,
        trainable_vars: list[str],
        non_trainable_vars: list[str],
        # constants: list[float],
    ) -> None:
        super().__init__()
        self.vparams = {p: torch.rand(1, requires_grad=True) for p in trainable_vars}
        self.fparams = {p: None for p in non_trainable_vars}
        # self.constants = {str(c): torch.tensor(c) for c in constants}
        self._dtype = torch.float64
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def to(self, args, kwargs) -> None:
        self.vparams = {p: t.to(*args, **kwargs) for p, t in self.vparams.items()}
        # self.constants = {c: t.to(*args, **kwargs) for c, t in self.constants.items()}
        try:
            k = next(iter(self.vparams))
            t = self.vparams[k]
            self._device = t.device
            self._dtype = t.dtype
        except Exception:
            pass

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
    """A class holding a parameterbuffer with concretized vparams a list of featureparams,
    and a list of "assigned parameters" which can be results of function/expression evaluations.
    """

    def __init__(
        self,
        param_buffer: ParameterBuffer,
        assigns_to_torch: dict[str, Callable[[dict, dict], torch.Tensor]],
    ) -> None:
        super().__init__()
        self.param_buffer = param_buffer
        self.assigns_to_torch = assigns_to_torch

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assigned_params = {}
        try:
            assert inputs.keys() == self.param_buffer.fparams.keys()
        except Exception as e:
            logger.error("Please pass a dict containing name:value for each fparam.")
        for assign_name, fn_or_value in self.assigns_to_torch.items():
            assigned_params[assign_name] = fn_or_value(
                self.param_buffer.vparams,
                {
                    **inputs,
                    **assigned_params,
                },  # we add the "intermediate" variables too
            )

        return assigned_params


def parse_assigns(model: Model) -> dict[str, Callable[[list[str]], torch.Tensor]]:
    assign_to_torch = dict()
    for instr in model.instructions:
        if isinstance(instr, Assign):
            assign_to_torch[instr.variable] = TorchCallable(instr.value)
    return assign_to_torch


def compile_circ(
    model: Model,
) -> pyq.QuantumCircuit:
    pyq_operations = []
    for instr in model.instructions:
        if isinstance(instr, QuInstruct):
            native_op = None
            try:
                native_op = getattr(pyq, instr.name.upper())
            except Exception as e:
                native_op = Q_OpMap[instr.name]
            control = instr.support.control
            target = instr.support.target
            native_support = (*control, *target)
            if len(instr.args) > 0:
                assert len(instr.args) == 1, "More than one arg not supported"
                (maybe_load,) = instr.args
                assert isinstance(maybe_load, Load), "only support load"
                pyq_operations.append(native_op(native_support, maybe_load.variable))
            else:
                pyq_operations.append(native_op(*native_support))
    return pyq.QuantumCircuit(model.register.num_qubits, pyq_operations)


def pyq_compile(model) -> (Embedding, pyq.QuantumCircuit):
    buffer = ParameterBuffer.from_model(model)
    torched_assigns = parse_assigns(model)
    embedding = Embedding(buffer, torched_assigns)
    native_circ = compile_circ(model)
    return embedding, native_circ
