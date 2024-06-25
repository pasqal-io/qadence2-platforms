from __future__ import annotations

from types import ModuleType
from copy import deepcopy
from typing import Any, Callable, Union

from qadence2_platforms.types import (
    BytecodeInstructType,
    SequenceObjectType,
    DeviceType,
    RegisterType,
)
from qadence2_platforms.backend.api import DialectAPI
from qadence2_platforms.backend.bytecode import Bytecode
from qadence2_platforms.backend.utils import (
    get_backend_instruct_instance,
    get_sequence_instance,
    get_register_instance,
    get_device_module,
)
from qadence2_platforms.qadence_ir import Alloc, Assign, Call, Load, QuInstruct, Model

instr_map = {
    "not": "not_fn",
    "z": "z_fn",
    "h": "h_fn",
    "rx": "rx_fn",
    "qubit_dyn": "qubit_dyn_fn"
}


class Dialect(
    DialectAPI[RegisterType, str, DeviceType, BytecodeInstructType, SequenceObjectType]
):
    """
    <Add the `Dialect` description here>

    It is an intermediate resolver class that must be called by the compile function.
    By invoking its `compile` method, it generates a `Bytecode` iterator instance, which
    is necessary for the runtime functionalities, such as `sample`, `expectation`, etc.,
    with arguments such as number of shots, feature parameters input, error mitigation
    options, result types, and so on.

    Here it is assumed that `Model`'s `inputs` attribute (a dictionary) will contain the
    feature parameters (data provided by the user), and, subsequently, the inputs and all
    the SSA form variables will be located in a single source, now called `var_dict`.
    """

    def __init__(
        self, backend: str, model: Model, device: str | None = None
    ):
        self.model: Model = model
        self.device_name: str = device
        self.backend_name: str = backend

        self.var_dict: dict[str, Union[Call, Alloc, Assign]] = deepcopy(self.model.inputs)
        self.device: DeviceType = get_device_module(
            backend=self.backend_name,
            device=self.device_name
        )

        register_fn: Callable = get_register_instance(self.backend_name)
        self.register: RegisterType = register_fn(self.model, self.device)

        self.backend_module: Callable = get_backend_instruct_instance(
            backend=self.backend_name,
            device=self.device_name
        )

        sequence_instance: Callable = get_sequence_instance(
            backend=self.backend_name,
            device=self.device_name
        )
        self.sequence: SequenceObjectType = sequence_instance(
            register=self.register,
            device=self.device,
            directives=self.model.directives
        )

    def _resolve_instructions(self) -> tuple[BytecodeInstructType, ...]:
        """
        It resolves instructions to place all the assignments to the `var_dict` attribute,
        while loading the appropriate sequence type and function to the `QuInstruct`
        instructions.

        :return: iterable of partial functions from instructions converted to backend's
         proper sequence type.
        """

        instr_iter: tuple[BytecodeInstructType, ...] = ()
        for instr in self.model.instructions:
            match instr:
                case Assign():
                    self.var_dict[instr.variable] = instr.value
                case QuInstruct():
                    instr_fn = getattr(self.backend_module, instr.name)
                    instr_iter += instr_fn(seq=self.sequence, support=instr.support, *instr.args),

        return instr_iter

    def compile(self) -> Bytecode:
        """
        It resolves `QuInstruct` into appropriate backend's sequence type, creates the
        appropriate backend's `Register` instance, addresses and converts the directives,
        sets the appropriate data settings, and generates the `Bytecode` instance.

        :return: the `Bytecode` instance.
        """

        instructions = self._resolve_instructions()
        return Bytecode(
            backend=self.backend_name,
            sequence=self.sequence,
            instructions=instructions,
            device=self.device_name,
        )
