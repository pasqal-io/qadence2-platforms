from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from types import ModuleType
from typing import Callable, Generic, Optional

from qadence2_platforms.backend.bytecode import BytecodeApi
from qadence2_platforms.backend.interface import RuntimeInterfaceApi
from qadence2_platforms.backend.sequence import SequenceApi
from qadence2_platforms.backend.utils import (
    get_backend_module,
    get_backend_register_fn,
    get_device_instance,
    get_embedding_instance,
    get_native_seq_instance,
)
from qadence_ir.ir import Model
from qadence2_platforms.types import DeviceType, EmbeddingType, RegisterType


class DialectApi(ABC, Generic[RegisterType, DeviceType, EmbeddingType]):
    """
    <Add the `Dialect` description here>

    It is an intermediate resolver class that must be called by the compile function.
    By invoking its `compile` method, it generates a `Bytecode` iterator instance, which
    is necessary for the runtime functionalities, such as `sample`, `expectation`, etc.,
    with arguments such as number of shots, feature parameters input, error mitigation
    options, result types, and so on.

    Here it is assumed that `Model`'s `inputs` attribute (a dictionary) will contain the
    feature parameters (data provided by the user), and, subsequently, the inputs and all
    the SSA form variables will be located in a single source, now called `embedding`.
    """

    def __init__(self, backend: str, model: Model, device: Optional[str] = None):
        self.model: Model = model
        self.device_name: str = device or ""
        self.backend_name: str = backend

        self.device: DeviceType = get_device_instance(
            backend=self.backend_name, device=self.device_name
        )

        register_fn: Callable = get_backend_register_fn(self.backend_name)
        self.register: RegisterType = register_fn(self.model, self.device)

        self.interface_backend: ModuleType = get_backend_module(
            backend=self.backend_name
        )
        self.native_backend: ModuleType = import_module(self.backend_name)

        embedding_instance: Callable = get_embedding_instance(self.backend_name)
        self.embedding: EmbeddingType = embedding_instance(self.model)

        native_seq_instance: Callable = get_native_seq_instance(
            backend=self.backend_name, device=self.device_name
        )
        self.native_sequence: SequenceApi = native_seq_instance(
            model=self.model, register=self.register, device=self.device
        )

    @abstractmethod
    def compile_bytecode(self) -> BytecodeApi:
        """
        It resolves `QuInstruct` into appropriate backend's sequence type, creates the
        appropriate backend's `Register` instance, addresses and converts the directives,
        sets the appropriate data settings, and generates the `Bytecode` instance.

        :return: the `Bytecode` instance.
        """
        raise NotImplementedError()

    @abstractmethod
    def compile(self) -> RuntimeInterfaceApi:
        """
        It resolves `Model` into an appropriate backend's runtime object. It must load
        the desired backend and device, if available, and use the backend's implementation
        to provide the correct interface for the expression to be converted into native
        instructions and thus be runnable into the backends specifications.

        :return: a `RuntimeInterface` instance.
        """
        raise NotImplementedError()
