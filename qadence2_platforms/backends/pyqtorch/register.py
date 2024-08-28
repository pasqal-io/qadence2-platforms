from __future__ import annotations


class RegisterInterface:
    def __init__(self, n_qubits: int, init_state: str | None = None) -> None:
        self.n_qubits = n_qubits
        self.init_state = init_state
