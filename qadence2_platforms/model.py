from __future__ import annotations

from typing import Any, Literal


class Parameter:
    """
    Symbolic parameter to be defined at run time. The type of the parameter is
    defined by the backend.

    Inputs:
        name: Unique parameter name.
        size: Space occuped by the parameter.
        trainable: Flag if the parameter can change during a training loop.
    """

    def __init__(self, name: str, size: int, *, trainable: bool) -> None:
        self.name = name
        self.size = size
        self.is_trainable = trainable

    def __repr__(self) -> str:
        mut_flag = "mut " if self.is_trainable else ""
        return f"{mut_flag}{self.name}"

    def __len__(self) -> int:
        return self.size


class Support:
    def __init__(
        self,
        *indices: int,
        target: tuple[int, ...] | None = None,
        control: tuple[int, ...] | None = None,
    ) -> None:
        if indices and (target or control):
            raise SyntaxError(
                "Please, provide either qubit indices or target-control tuples"
            )

        if control and not target:
            raise SyntaxError("A controlled operation needs both, control and target.")

        if indices:
            self.target: tuple[int, ...] = indices
            self.control: tuple[int, ...] = ()
        else:
            self.target = target or ()
            self.control = control or ()

    @classmethod
    def all(cls) -> Support:
        return Support()


class Instruction:
    """
    An abstract representation of backend instruction.

    Inputs:
        name: The instruction name compatible with the standard instruction set.
        support: The index of qubits to which the instruction is applied to.
        args: Arguments of the instruction such as angle, duration, amplitude etc.
    """

    def __init__(self, name: str, support: Support, *args: Any):
        self.name = name
        self.support = support
        self.args = args

    def __repr__(self) -> str:
        return f"@{self.name} {self.support} ({', '.join(str(k) for k in self.args)})"


class Register:
    """
    Describes the atomic configuration of the register in a neutral atoms device.

    Inputs:
        qubits_positions: A list of coordinates in a discrete grid been (0,0) the
            center of the grid.
        grid_type: Allows to select the coordinates sets for the grid: "square"
            (orthogonal) or "triagular" (skew)
        grid_scale: Adjust the distance between atoms based on a standard distance
            defined by the backend
        options: Extra register related properties that may not be supported by
            all backends.
    """

    def __init__(
        self,
        qubits_positions: list[tuple[int, int]],
        grid_type: Literal["square", "triangular"],
        grid_scale: float = 1.0,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.qubits_positions = qubits_positions
        self.grid_type = grid_type
        self.grid_scale = grid_scale
        self.options = options or dict()


class Model:
    """
    Aggregates the minimal information to construct sequence of instructions in
    a quantum device. The structure is mainly focused in neutral atoms devices
    but its agnostic nature may make it suitable for any quantum device.

    Inputs:
        register: Describe the atomic arragement of the neutal atom register.
        instructions:  A list of abstract instructions with their arguments with
            which a backend can execute a sequence.
        directives: A dictionary containing QPU related options. For instance,
            it can be used to set the Rydberg level to be used or whether or not
            allow digital-analog operations in the sequence.
        backend_settings: Backend specific configurations (mostly for simulators)
            where the user can define for instance, the data type like `int64`,
            or the return type as "counting", "vector-state" or "density-matrix".
    """

    def __init__(
        self,
        register: Register,
        instructions: list[Instruction],
        directives: dict[str, Any] | None = None,
        backend_settings: dict[str, Any] | None = None,
    ) -> None:
        self.register = register
        self.instr = instructions
        self.directives = directives or dict()
        self.backend_settigns = backend_settings or dict()