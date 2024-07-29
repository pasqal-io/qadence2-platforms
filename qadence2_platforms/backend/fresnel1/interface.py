from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Literal
from qutip import Qobj

from pulser.sequence.sequence import Sequence
from pulser_simulation.simulation import QutipEmulator

from qadence2_platforms import AbstractInterface


class Interface(AbstractInterface[Sequence, float, Counter | Qobj]):

    def __init__(
        self,
        sequence: Sequence,
    ) -> None:
        self._params: dict[str, float] = {}
        self._sequence = sequence

    @property
    def info(self) -> dict[str, Any]:
        return {"device": self.sequence.device, "register": self.sequence.register}

    @property
    def sequence(self) -> Sequence:
        return self._sequence

    def set_parameters(self, params: dict[str, float]) -> None:
        valid_parms = params.keys() & self.sequence.declared_variables.keys()

        if valid_parms != params.keys():
            raise ValueError(
                "The sequence does not have the parameters {set(params.keys())}"
            )

        self._params = params

    def run(
        self,
        *,
        values: dict[str, float] | None = None,
        shots: int | None = None,
        callback: Callable | None = None,
        on: Literal["emulator", "qpu"] = "emulator",
        **kwargs: Any,
    ) -> Counter | Qobj:
        vals: dict[str, float] = {**(values or {}), **self._params}

        match on:
            case "emulator":
                built_sequence = self.sequence.build(**vals)  # type: ignore
                simulation = QutipEmulator.from_sequence(
                    built_sequence, with_modulation=True
                )
                result = simulation.run()
                if shots:
                    return result.sample_final_state(N_samples=shots)
                return result.get_final_state()
            case _:
                raise NotImplementedError
