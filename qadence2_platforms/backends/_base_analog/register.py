from __future__ import annotations

from typing import Any, Union, Literal

import numpy as np
from numpy.typing import ArrayLike
from pulser import AnalogDevice
from pulser.devices import Device
from pulser.register import RegisterLayout

from qadence2_platforms.backends._base_analog.device_settings import DeviceSettings

qubits_pos_type = list[tuple[int, int]]
coords_type = Union[ArrayLike, list[ArrayLike], tuple[ArrayLike]]


class RegisterTransform:
    """Transforms register data according to the `grid_type` in the `qadence2_ir.types.Model`"""

    _grid: Literal["linear", "triangular", "square"]
    _grid_scale: float
    _raw_coords: coords_type
    _transformed_coords: coords_type
    _device: Device

    def __init__(
        self,
        grid_transform: Literal["linear", "triangular", "square"],
        grid_scale: float = 1.0,
        coords: qubits_pos_type | None = None,
        num_qubits: int | None = None,
        device_settings: DeviceSettings | None = None,
    ):
        """
        Args:
            grid_transform (Literal["linear", "triangular", "square"]: literal str to choose
                which grid transform to use. Accepted values are "linear", "triangular"
                or "square"
            grid_scale (float): scale of the grid. Default is `1.0`
            coords (list[tuple[int, int]]): list of coordinates as qubit positions in an int
                grid, e.g. `[(0, 0), (1, 0), (0, 1)]`. Default is `None`
            num_qubits (int | None): number of qubits as integer. Default is `None`
            device_settings (DeviceSettings | None): Device settings object
        """

        self._grid = grid_transform
        self._grid_scale = grid_scale

        if coords:
            self._raw_coords = coords

        elif num_qubits:
            self._raw_coords = self._fill_coords(num_qubits)

        else:
            raise ValueError("must provide coords or num_qubits.")

        print(f"{hasattr(self, f'{self._grid}_coords')=}")
        self._transformed_coords = getattr(
            self,
            f"{self._grid}_coords",
            # self.invalid_grid_value()
        )()
        self._device = device_settings.device

    @property
    def raw_coords(self) -> coords_type:
        return self._raw_coords

    @property
    def coords(self) -> coords_type:
        return self._transformed_coords

    @property
    def device(self) -> Device:
        return self._device

    @classmethod
    def _fill_coords(cls, num_qubits: int) -> coords_type:
        shift = num_qubits // 2
        return [(p - shift, 0) for p in range(num_qubits)]

    def invalid_grid_value(self) -> None:
        """Fallback function for invalid `grid_transform` value."""
        print(f"{self._grid=}")
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
        print(f"triangular coords")
        # triangular transformation matrix
        transform = np.array([[1.0, 0.0], [0.5, 0.8660254037844386]])
        return np.array(self._raw_coords) * self._grid_scale @ transform

    def square_coords(self) -> np.ndarray:
        """
        Transforms coordinates into square coordinates.

        Returns:
            np.ndarray of transformed coordinates
        """

        # for now, no transformation needed since the coords are list of tuple of ints
        return np.array(self._raw_coords) * self._grid_scale

    def get_calibrated_layout(self, layout_name: str) -> RegisterLayout:
        return AnalogDevice.calibrated_register_layouts.get(layout_name, None)
