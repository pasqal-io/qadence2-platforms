from __future__ import annotations

from qadence2_platforms.qadence_ir import AllocQubits


class RegisterInterface:
    def __init__(self, register: AllocQubits):
        self.n_qubits: int = register.num_qubits
        self.init_state: str | None = register.options.get("init_state", None)
