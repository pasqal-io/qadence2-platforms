from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np
from pulser.channels import DMM
from pulser.devices import AnalogDevice

from qadence2_platforms.backends._base_analog.device_settings import DeviceSettings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.formatwarning = lambda msg, *args, **kwargs: f"WARNING: {msg}\n"


class Fresnel1DeviceSettings(DeviceSettings):
    """Defines Fresnel-1 specific settings for checks."""

    def __init__(self) -> None:
        self._name = "Fresnel-1"
        self._name_short = "fresnel1"
        self._device = replace(
            AnalogDevice.to_virtual(),
            dmm_objects=(
                DMM(
                    # from Pulser tutorials/dmm.html#DMM-Channel-and-Device
                    clock_period=4,
                    min_duration=16,
                    max_duration=2**26,
                    mod_bandwidth=8,
                    bottom_detuning=-2 * np.pi * 20,  # detuning between 0 and -20 MHz
                    total_bottom_detuning=-2 * np.pi * 2000,  # total detuning
                ),
            ),
        )
        self._grid_scale_range = (1.0, 1.0)
        self._available_grid_types = ("triangular",)
        # TODO: check which directives should or must be present
        self._available_directives = ()


# Define the Fresnel-1 settings, including the device object and
# its limitations or parameters range.
Fresnel1Settings = Fresnel1DeviceSettings()
