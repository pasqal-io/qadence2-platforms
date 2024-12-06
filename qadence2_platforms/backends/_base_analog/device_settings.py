from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pulser.devices import Device


class DeviceSettings(ABC):
    """
    Device settings for easy of building sequence, register and interface logic.

    name (str): the name of the device, ex: `Analog Device`
    name_short (str): a shorter name version, ex: `analog`
    device (pulser.devices.Device): the pulser device
    grid_scale_range (tuple[float, float]): a tuple of min and max values for the
        grid scale, ex: `(1.0, 1.0)` (should not scale), `(1.0, 10.0)` (up to
        10.0, included)
    available_grid_types (tuple[str]): a tuple of all the possible grid types
        for the device, ex: `("triangular",)`, `("linear", "square")`
    available_directives (tuple[str], optional): a tuple of available directives,
        ex: `("enable_digital_analog")`
    """

    _name: str
    _name_short: str
    _device: Device
    _grid_scale_range: tuple[float, float]
    _available_grid_types: tuple[str, ...]
    _available_directives: tuple[str, ...] | tuple | None

    @property
    def name(self) -> str:
        return self._name

    @property
    def name_short(self) -> str:
        return self._name_short

    @property
    def device(self) -> Device:
        return self._device

    @property
    def grid_scale_range(self) -> tuple[float, float]:
        return self._grid_scale_range

    @property
    def available_grid_types(self) -> tuple[str, ...]:
        return self._available_grid_types

    @property
    def available_directives(self) -> tuple[str, ...] | tuple | None:
        return self._available_directives

    def scale_in_range(self, grid_scale: float):
        return self._grid_scale_range[0] <= grid_scale <= self._grid_scale_range[1]

    @abstractmethod
    def check_grid_scale(self, grid_scale: float) -> Any:
        pass

    @abstractmethod
    def check_grid_type(self, grid_type: str) -> Any:
        pass

    @abstractmethod
    def check_directives(self, directives: Any) -> Any:
        pass
