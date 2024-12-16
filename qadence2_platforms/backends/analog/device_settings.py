from __future__ import annotations

from pulser.devices import AnalogDevice

from qadence2_platforms.backends._base_analog.device_settings import DeviceSettings


class AnalogDeviceSettings(DeviceSettings):
    """Defines AnalogDevice specific settings for checks"""

    def __init__(self) -> None:
        self._name = "AnalogDevice"
        self._short_name = "analog"
        self._device = AnalogDevice
        self._grid_scale_range = (1.0, 100.0)
        # TODO: validate whether the "square" grid type is working properly
        self._available_grid_types = ("triangular", "square")
        self._available_directives = ("local_shifts", "local_targets", "enable_digital_analog")


# Define the AnalogDevice settings, including device object and
# its limitations or parameters range.
AnalogSettings = AnalogDeviceSettings()
