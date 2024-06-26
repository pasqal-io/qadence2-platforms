from __future__ import annotations

from typing import TypeVar

Scalar = float

# model generic types
RegisterType = TypeVar("RegisterType")
DeviceType = TypeVar("DeviceType")
DirectivesType = TypeVar("DirectivesType")
QuInstructType = TypeVar("QuInstructType")
SequenceObjectType = TypeVar("SequenceObjectType")

# instructions generic types
SupportType = TypeVar("SupportType")
ArgsType = TypeVar("ArgsType")

# backend generic types
BackendInstructResultType = TypeVar("BackendInstructResultType")
BackendType = TypeVar("BackendType")

# bytecode generic types
BytecodeInstructType = TypeVar("BytecodeInstructType")
UserInputType = TypeVar("UserInputType")

# interface generic types
InterfaceInstructType = TypeVar("InterfaceInstructType")
RunResultType = TypeVar("RunResultType")
SampleResultType = TypeVar("SampleResultType")
ExpectationResultType = TypeVar("ExpectationResultType")
InterfaceCallResultType = TypeVar("InterfaceCallResultType")


class BackendName:
    PULSER = "pulser"
    PYQTORCH = "pyqtorch"
    HORQRUX = "horqrux"


class DeviceName:
    FRESNEL = "Fresnel"
    FRESNEL_EOM = "FresnelEOM"
    ANALOG_DEVICE = "AnalogDevice"
    NONE = ""
