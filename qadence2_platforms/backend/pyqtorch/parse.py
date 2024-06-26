from __future__ import annotations

from typing import Callable

import pyqtorch as pyq
import torch

from qadence2_platforms.qadence_ir import (
    Assign,
    Call,
    Load,
    QuInstruct,
)

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


def TorchCallable(call: Call) -> Callable[[list[str], torch.Tensor]]:
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
    def __init__(
        self,
        trainable_vars: list[str],
        non_trainable_vars: list[str],
        constants: list[float],
    ) -> None:
        super().__init__()
        self.vparams = {p: torch.rand(1, requires_grad=True) for p in trainable_vars}
        self.fparams = {p: None for p in non_trainable_vars}
        self.constants = {str(c): torch.tensor(c) for c in constants}
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
        self.constants = {c: t.to(*args, **kwargs) for c, t in self.constants.items()}
        try:
            k = next(iter(self.vparams))
            t = self.vparams[k]
            self._device = t.device
            self._dtype = t.dtype
        except Exception:
            pass


class Embedding(torch.nn.Module):
    def __init__(self, fns: list[Call]) -> None:
        super().__init__()
        self.fn_and_inputs = {getattr(torch, fn.call) for fn in fns}

    def __call__(
        self, param_buffer: ParameterBuffer, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        pass


def parse_instruction(
    instr: QuInstruct | Assign | Load | Call | list,
) -> pyq.primitive.Primitve | pyq.QuantumCircuit | torch.nn.Module:
    c_ops, q_ops = {}, []
    if isinstance(instr, list):
        return list(map(parse_instruction, instr))
    elif isinstance(instr, QuInstruct):
        native_op = None
        try:
            native_op = getattr(pyq, instr.name.upper())
        except Exception as e:
            native_op = Q_OpMap[instr.name]
        control = instr.support.control
        target = instr.support.target
        native_support = (*control, *target)
        if len(instr.args) > 0:
            params = list(map(parse_instruction, instr.args))
            return q_ops.append(native_op(native_support, *params))
        else:
            return q_ops.append(native_op(*native_support))
    elif isinstance(instr, Assign):
        c_ops[instr.variable] = parse_instruction(instr.value)
    elif isinstance(instr, Load):
        return instr.variable
    elif isinstance(instr, Call):
        return TorchCallable(instr)

    elif isinstance(instr, (str, float)):
        return instr

    else:
        raise NotImplementedError(f"Not supported operation: {instr}")
