from __future__ import annotations

from dataclasses import replace
from collections import Counter
from unittest.mock import Base
import numpy as np

from pulser.channels import DMM
from pulser.devices._devices import AnalogDevice
from pulser.devices._device_datacls import BaseDevice
from pulser.register.register_layout import RegisterLayout
from pulser.sequence.sequence import Sequence
from pulser_simulation.simulation import QutipEmulator

from qadence2_platforms import AbstractInterface

# Fresnel1 = AnalogDevice
Fresnel1 = replace(
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
    )
)


class Interface(AbstractInterface[Sequence, Counter | np.ndarray]):

    def __init__(self, register: RegisterLayout) -> None:
        self._register = register
    
    @property
    def register(self) -> RegisterLayout:
        return self._register
    
