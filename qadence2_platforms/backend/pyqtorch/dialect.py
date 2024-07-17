from __future__ import annotations

from qadence2_platforms.backend.dialect import DialectApi

from .bytecode import Bytecode
from .embedding import EmbeddingModule
from .interface import RuntimeInterface
from .register import RegisterInterface


class Dialect(DialectApi[RegisterInterface, None, EmbeddingModule]):
    def compile_bytecode(self) -> Bytecode:
        raise NotImplementedError()

    def compile(self) -> RuntimeInterface:
        native_seq_compiled = self.native_sequence.build_sequence()
        return RuntimeInterface(
            register=self.register,
            embedding=self.embedding,
            native_seq=native_seq_compiled,
            native_backend=self.native_backend,
            observable=None,
        )
