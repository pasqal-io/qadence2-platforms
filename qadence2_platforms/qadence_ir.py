from __future__ import annotations

from typing import Any, Literal


class Alloc:
    """
    Reserve one slot for a scaler parameter in the environment and n-slots for
    an array. The type of the parameter is defined by the backend.

    Inputs:
        size: Space occuped by the parameter.
        trainable: Flag if the parameter can change during a training loop.
    """

    def __init__(self, size: int, trainable: bool) -> None:
        self.size = size
        self.is_trainable = trainable


class Assign:
    """Push a variable to the environment and assign a value to it."""

    def __init__(self, variable_name: str, value: Any) -> None:
        self.variable = variable_name
        self.value = value


class Load:
    """To recover the value of a given variable."""

    def __init__(self, variable_name: str) -> None:
        self.variable = variable_name


class Call:
    """
    Indicates the call of classical functions only.
    """

    def __init__(self, function_name: str, *args: Any) -> None:
        self.call = function_name
        self.args = args


class Support:
    """
    Generic representation of the qubit support. For single qubit operations,
    a muliple index support indicates apply the operation for each index in the
    support.

    Both target and control lists must be ordered!

    Inputs:
       target = Index or indices where the operation is applied.
       control = Index or indices to which the operation is conditioned to.
    """

    def __init__(
        self,
        *,
        target: tuple[int, ...],
        control: tuple[int, ...] | None = None,
    ) -> None:
        self.target = target
        self.control = control or ()

    @classmethod
    def target_all(cls) -> Support:
        return Support(target=())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Support):
            return NotImplemented

        return self.target == other.target and self.control == other.control


class QuInstruct:
    """
    An abstract representation of a QPU instruction.

    Inputs:
        name: The instruction name compatible with the standard instruction set.
        support: The index of qubits to which the instruction is applied to.
        args: Arguments of the instruction such as angle, duration, amplitude etc.
    """

    def __init__(self, name: str, support: Support, *args: Any):
        self.name = name
        self.support = support
        self.args = args


class AllocQubits:
    """
    Describes the register configuration in a neutral atoms device.

    Inputs:
        num_qubits: Number of atoms to be allocated.
        qubit_positions: A list of discrete coordinates for 2D grid with (0,0)
            position at center of the grid. A list of indices in a linear register.
            An empty list will indicate the backen is free to define the topology
            for devices that implement logical qubits.
        grid_type: Allows to select the coordinates sets for 2D grids: "square"
            (orthogonal) or "triagular" (skew). A "linear" will allow the backend
            to define the shape of the register. When the `grid_type` is `None`
            the backend uses its default structure (particular useful when
            shuttling is available). Default value is `None`.
        grid_scale: Adjust the distance between atoms based on a standard distance
            defined by the backend. Default value is 1.
        options: Extra register related properties that may not be supported by
            all backends.
    """

    def __init__(
        self,
        num_qubits: int,
        qubit_positions: list[tuple[int, int]] | list[int],
        grid_type: Literal["linear", "square", "triangular"] | None = None,
        grid_scale: float = 1.0,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.qubit_positions = qubit_positions
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
        data_settings: Backend specific configurations where the user can define
            for instance, the data type like `int64`, or the return type as
            "counting", "vector-state" or "density-matrix".
    """

    def __init__(
        self,
        register: AllocQubits,
        inputs: dict[str, Assign],
        instructions: list[QuInstruct | Assign],
        directives: dict[str, Any] | None = None,
        data_settings: dict[str, Any] | None = None,
    ) -> None:
        self.register = register
        self.inputs = inputs
        self.instructions = instructions
        self.directives = directives or dict()
        self.data_settings = data_settings or dict()