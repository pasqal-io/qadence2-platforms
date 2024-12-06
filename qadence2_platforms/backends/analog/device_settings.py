from __future__ import annotations

from pulser.devices import AnalogDevice

from qadence2_platforms.backends._base_analog.device_settings import DeviceSettings


AnalogDeviceSettings = DeviceSettings(
    name="AnalogDevice",
    name_short="analog",
    device=AnalogDevice,
    available_grid_types=("linear", "triangular", "square"),
)
