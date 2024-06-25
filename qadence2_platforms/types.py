from __future__ import annotations

from typing import TypeVar


Scalar = float

# generic types for backend purposes
RegisterType = TypeVar("RegisterType")
DeviceType = TypeVar("DeviceType")
DirectivesType = TypeVar("DirectivesType")
QuInstructType = TypeVar("QuInstructType")
SequenceObjectType = TypeVar("SequenceObjectType")
BackendType = TypeVar("BackendType")
BytecodeInstructType = TypeVar("BytecodeInstructType")


class BackendName:
    PULSER = "pulser"
    PYQTORCH = "pyqtorch"
    HORQRUX = "horqrux"


class DeviceName:
    FRESNEL = "fresnel"
    FRESNEL_EOM = "fresnel_eom"
    ANALOG_DEVICE = "analog_device"
    NONE = ""
