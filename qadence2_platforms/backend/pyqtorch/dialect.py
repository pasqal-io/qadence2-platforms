from __future__ import annotations

from functools import cached_property

from qadence2_platforms.backend.dialect import DialectApi

from .bytecode import Bytecode
from .embedding import EmbeddingModule
from .interface import RuntimeInterface


class Dialect(DialectApi):
    @cached_property
    @property
    def embedding(self) -> EmbeddingModule:
        return EmbeddingModule(self.model)

    def compile_bytecode(self) -> Bytecode:
        raise NotImplementedError()

    def compile(self) -> RuntimeInterface:

        return RuntimeInterface(
            register=self.register,
            embedding=self.embedding,
            native_seq=self.native_sequence,
            native_backend=self.native_backend,
            observable=None,
        )
