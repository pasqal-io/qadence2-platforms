from __future__ import annotations

from functools import cached_property

from pulser.register.base_register import BaseRegister
from pulser.devices._device_datacls import BaseDevice

from qadence2_platforms.backend.dialect import DialectApi

from .bytecode import Bytecode
from .embedding import EmbeddingModule
from .interface import RuntimeInterface


class Dialect(DialectApi[BaseRegister, BaseDevice]):
    @cached_property
    @property
    def embedding(self) -> EmbeddingModule:
        return EmbeddingModule(self.model)

    def compile_bytecode(self) -> Bytecode:
        raise NotImplementedError()

    def compile(self) -> RuntimeInterface:
        raise NotImplementedError()
