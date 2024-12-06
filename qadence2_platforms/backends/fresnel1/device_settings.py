from __future__ import annotations

from dataclasses import replace
from typing import Any
import warnings

import numpy as np
from pulser.channels import DMM
from pulser.devices import AnalogDevice

from qadence2_platforms.backends._base_analog.device_settings import DeviceSettings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.formatwarning = lambda msg, *args, **kwargs: f"WARNING: {msg}\n"


class Fresnel1DeviceSettings(DeviceSettings):
    """Defines Fresnel-1 specific settings and checks."""

    def __init__(self):
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
        self._available_directives = ()

    def check_grid_scale(self, grid_scale: float) -> Any:
        if self.scale_in_range(grid_scale):
            warnings.warn(
                "Currently, Fresnel-1 uses a fixed grid.",
                SyntaxWarning,
                stacklevel=2,
            )

    def check_grid_type(self, grid_type: str) -> Any:
        if grid_type and grid_type not in Fresnel1Settings.available_grid_types:
            warnings.warn(
                "Fresnel-1 only supports triangular grids at the moment.",
                SyntaxWarning,
                stacklevel=2,
            )

    def check_directives(self, directives: dict[str, Any]) -> Any:
        if directives.get("enable_digital_analog"):
            warnings.warn(
                "Fresnel-1 uses a fixed grid and does not have digital channels.\n"
                + "Digital operations using atomic distance-based strategies\n"
                + "may not behave as expected.",
                SyntaxWarning,
                stacklevel=2,
            )


Fresnel1Settings = Fresnel1DeviceSettings()
