from __future__ import annotations

from typing import Union, Callable, Any

import numpy as np
from numpy.typing import ArrayLike
from pulser import AnalogDevice
from pulser.devices import Device
from pulser.register import RegisterLayout
from qadence2_ir.types import Model

from qadence2_platforms.backends._base_analog.device_settings import DeviceSettings
from qadence2_platforms.backends.utils import gridtype_literal

qubits_pos_type = list[tuple[int, int]]
coords_type = Union[ArrayLike, list[ArrayLike], tuple[ArrayLike]]


class RegisterTransform:
    """Transforms register data according to the `grid_type` in the `qadence2_ir.types.Model`"""

    _device: Device
    _grid: gridtype_literal
    _grid_scale: float
    _raw_coords: coords_type
    coords: coords_type

    # below is a multiplying factor that seems to be needed to correct the values coming
    # from the `qadence2-expressions`'s `qubit_positions`. It should be used on the
    # coordinate transformations
    scale_factor: int = 5

    def __init__(
        self,
        device_settings: DeviceSettings,
        grid_transform: gridtype_literal | None,
        grid_scale: float = 1.0,
        coords: qubits_pos_type | None = None,
        num_qubits: int | None = None,
    ):
        """
        Args:
            grid_transform (Literal["linear", "triangular", "square"], None): literal str to choose
                which grid transform to use. Accepted values are "linear", "triangular"
                or "square". If None is provided, it will default to "triangular"
            grid_scale (float): scale of the grid. Default is `1.0`
            coords (list[tuple[int, int]]): list of coordinates as qubit positions in an int
                grid, e.g. `[(0, 0), (1, 0), (0, 1)]`. Default is `None`
            num_qubits (int | None): number of qubits as integer. Default is `None`
            device_settings (DeviceSettings | None): Device settings object
        """

        self._grid = grid_transform if grid_transform is not None else "triangular"
        self._grid_scale = grid_scale

        if coords:
            self._raw_coords = coords

        elif num_qubits:
            self._raw_coords = self._fill_coords(num_qubits)

        else:
            raise ValueError("must provide coords or num_qubits.")

        try:
            self.coords = getattr(self, f"{self._grid}_coords")()
        except AttributeError:
            self.invalid_grid_value()
        else:
            self._device = device_settings.device

    @property
    def raw_coords(self) -> coords_type:
        return self._raw_coords

    @property
    def device(self) -> Device:
        return self._device

    @classmethod
    def _fill_coords(cls, num_qubits: int) -> coords_type:
        shift = num_qubits // 2
        return [(p - shift, 0) for p in range(num_qubits)]

    @classmethod
    def invalid_grid_value(cls) -> None:
        """Fallback function for invalid `grid_transform` value."""

        raise ValueError("grid_transform should be 'linear', 'triangular', or 'square'.")

    def linear_coords(self) -> np.ndarray:
        """
        Transforms coordinates into linear coordinates.

        Returns:
            np.ndarray of transformed coordinates.
        """

        raise NotImplementedError()

    def triangular_coords(self) -> np.ndarray:
        """
        Transforms coordinates into triangular coordinates.

        Returns:
            np.ndarray of transformed coordinates
        """

        # triangular transformation matrix
        transform = np.array([[1.0, 0.0], [0.5, 0.8660254037844386]])
        return np.array(self._raw_coords) * self._grid_scale * self.scale_factor @ transform

    def square_coords(self) -> np.ndarray:
        """
        Transforms coordinates into square coordinates.

        Returns:
            np.ndarray of transformed coordinates
        """

        # for now, no transformation needed since the coords are list of tuple of ints
        return np.array(self._raw_coords) * self._grid_scale * self.scale_factor

    @classmethod
    def get_calibrated_layout(cls, layout_name: str) -> RegisterLayout:
        """
        Gets the calibrated layout for the given `layout_name` according to `AnalogDevice`
        specifications.

        Args:
            layout_name: the name of the layout.

        Returns:
            The `RegisterLayout` object for the given `layout_name`.
        """

        return AnalogDevice.calibrated_register_layouts.get(layout_name, None)


class RegisterResolver:
    """
    An object to hold common functionalities for registers and resolving their
    layouts, transformations and checks.
    """

    @classmethod
    def resolve_from_model(
        cls,
        model: Model,
        device_settings: DeviceSettings,
        grid_scale_fn: Callable[[Model], Any],
        grid_type_fn: Callable[[Model], Any],
        directives_fn: Callable[[Model], Any],
        register_transform_fn: Callable[[RegisterTransform], RegisterLayout],
    ) -> RegisterLayout:
        """
        Resolves the model's register data into actual platform appropriate and
        validated data through platform's own functions. It is a generic and
        modular helper function to be called by `register.from_model` function.

        Args:
            model (Model): The model to use and resolve its data
            device_settings (DeviceSettings): The device settings instance
            grid_scale_fn (Callable): Function used to check the grid scale
                against the device's
            grid_type_fn (Callable): Function used to check the grid type
                against the device's
            directives_fn (Callable): Function used to check the directives
                against the device's
            register_transform_fn (Callable): Function used to transform
                coordinates into register on appropriate layout

        Returns:
            A register data with appropriate layout
        """

        model_register = model.register

        if model_register.num_qubits < 1:
            raise ValueError("No qubit available in the register.")

        grid_scale_fn(model)
        grid_type_fn(model)
        directives_fn(model)

        register_transform = RegisterTransform(
            grid_transform=model_register.grid_type,
            grid_scale=model_register.grid_scale,
            coords=model_register.qubit_positions,
            num_qubits=model_register.num_qubits,
            device_settings=device_settings,
        )

        register = register_transform_fn(register_transform)

        return register
