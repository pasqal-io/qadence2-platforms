from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from qadence2_platforms import Model
from qadence2_platforms.backend.embedding import EmbeddingModuleApi, ParameterBufferApi
from qadence2_platforms.qadence_ir import Assign, Call, Load

np_fun_map = {
    "mul": "multiply",
    "add": "sum",
}


def np_call(call: Call) -> Callable[[dict, dict], np.ndarray]:
    """Convert a `Call` object into a numpyfied function which can be evaluated using
    a vparams and inputs dict.
    """
    fn = getattr(np, call.call, None)
    if fn is None:
        fn = getattr(np, np_fun_map[call.call])

    def evaluate(params: dict, inputs: dict) -> np.ndarray:
        args = []
        for symbol in call.args:
            if isinstance(symbol, float):
                # NOTE we compile constants into each TorchCallable instead of passing
                # them around in the values dict
                args.append(np.array(symbol))
            elif isinstance(symbol, Load):
                args.append({**params, **inputs}[symbol.variable])
        return fn(*args)

    return evaluate


class ParameterBuffer(ParameterBufferApi[np.dtype, np.ndarray]):
    def __init__(self, trainable_vars: list[str], non_trainable_vars: list[str]):
        self.vparams: dict[str, np.ndarray] = {
            p: np.random.rand(1) for p in trainable_vars
        }
        self.fparams: dict[str, Optional[np.ndarray]] = {
            p: None for p in non_trainable_vars
        }
        self._dtype: np.dtype = np.float64

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


NameMappingType = dict[str, Callable[[dict, dict], np.ndarray]]


class EmbeddingModule(EmbeddingModuleApi[np.ndarray, NameMappingType]):
    def __init__(self, model: Model):
        self.model: Model = model
        self.param_buffer: ParameterBuffer = ParameterBuffer.from_model(self.model)
        self.mapped_vars: NameMappingType = self.name_mapping()

    def __call__(
        self, inputs: Optional[dict[str, np.ndarray]] = None
    ) -> dict[str, np.ndarray]:
        inputs = inputs or dict()
        assigned_params: dict[str, np.ndarray] = dict()
        assert (
            inputs.keys() == self.param_buffer.fparams.keys()
        ), "Please pass a dict containing name:value for each fparam."
        for var, npcall in self.mapped_vars.items():
            assigned_params[var] = npcall(
                self.param_buffer.vparams,
                {
                    **inputs,
                    **assigned_params,
                },  # we add the "intermediate" variables too
            )

        return assigned_params

    def name_mapping(self) -> NameMappingType:
        assign_values = dict()
        for instr in self.model.instructions:
            if isinstance(instr, Assign):
                assign_values[instr.variable] = np_call(instr.value)
        return assign_values
