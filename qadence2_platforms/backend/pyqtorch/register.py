from __future__ import annotations

from qadence2_platforms.qadence_ir import AllocQubits


class RegisterInterface:
    def __init__(self, model: AllocQubits):
        self.n_qubits: int = model.num_qubits
        self.init_state: str | None = model.options.get("init_state", None)
