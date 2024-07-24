from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Literal
import qutip as qt

from pulser.sequence.sequence import Sequence
from pulser_simulation.simulation import QutipEmulator

from qadence2_platforms import AbstractInterface


class Interface(AbstractInterface[Sequence, float, Counter | qt.Qobj]):

    def __init__(
        self,
        sequence: Sequence,
    ) -> None:
        self._sequence = sequence

    @property
    def info(self) -> dict[str, Any]:
        return {"device": self.sequence.device, "register": self.sequence.register}

    @property
    def sequence(self) -> Sequence:
        return self._sequence

    def add_noise(self, model: Literal["SPAM"]) -> None:
        pass

    def run(
        self,
        *,
        parameters: dict[str, float] | None = None,
        shots: int | None = None,
        callback: Callable | None = None,
        on: Literal["emulator", "qpu"] = "emulator",
        **kwargs: Any,
    ) -> Counter | qt.Qobj:
        match on:
            case "emulator":
                built_sequence = self.sequence.build(**(parameters or {}))  # type: ignore
                simulation = QutipEmulator.from_sequence(
                    built_sequence, with_modulation=True
                )
                result = simulation.run()
                if shots:
                    return result.sample_final_state(N_samples=shots)
                return result.get_final_state()
            case _:
                raise NotImplementedError
