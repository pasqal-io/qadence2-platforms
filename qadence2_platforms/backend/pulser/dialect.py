from __future__ import annotations

from pulser.devices._device_datacls import BaseDevice
from pulser.register.base_register import BaseRegister

from qadence2.platforms.backend.dialect import DialectApi

from .bytecode import Bytecode
from .embedding import EmbeddingModule
from .interface import RuntimeInterface


class Dialect(DialectApi[BaseRegister, BaseDevice, EmbeddingModule]):
    def compile_bytecode(self) -> Bytecode:
        raise NotImplementedError()

    def compile(self) -> RuntimeInterface:
        native_seq_compiled = self.native_sequence.build_sequence()
        return RuntimeInterface(
            register=self.register,
            embedding=self.embedding,
            native_seq=native_seq_compiled,
            native_backend=self.native_backend,
        )
