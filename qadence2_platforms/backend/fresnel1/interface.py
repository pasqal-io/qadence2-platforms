from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Literal, Optional, Union

from pulser.sequence.sequence import Sequence
from pulser_simulation.simresults import SimulationResults
from pulser_simulation.simulation import QutipEmulator
from qutip import Qobj

from qadence2_platforms import AbstractInterface

RunResult = Union[Counter, Qobj]


class Interface(AbstractInterface[float, Sequence, float, RunResult, Counter, Qobj]):

    def __init__(self, sequence: Sequence, non_trainable_parameters: set[str]) -> None:
        self._non_trainable_parameters = non_trainable_parameters
        self._params: dict[str, float] = {}
        self._sequence = sequence

    @property
    def info(self) -> dict[str, Any]:
        return {"device": self.sequence.device, "register": self.sequence.register}

    @property
    def sequence(self) -> Sequence:
        return self._sequence

    def set_parameters(self, params: dict[str, float]) -> None:
        valid_params = params.keys() & self._non_trainable_parameters

        if valid_params != params.keys():
            raise ValueError(
                f"{set(params.keys())} are not fixed parameters in this sequence."
            )

        self._params = params

    def _run(
        self,
        run_type: Literal["run", "sample", "expectation"],
        platform: SimulationResults,
        shots: Optional[int] = None,
        observable: Optional[Any] = None,
        callback: Optional[Callable] = None,
        **_: Any,
    ) -> Any:
        """
        Method to be used for `_on_emulator` or `_on_qpu` to execute the respective
        `run_type` option.

        **Notice**: for now, it only supports `emulator` option and
        `QutipEmulator` platform.

        :param run_type: str: `run`, `sample`, `expectation` options
        :param platform: callable to retrieve methods for executing the options above
        :param shots: number of shots, if applicable (`sample` only)
        :param observable: list of observables, if applicable (`expectation` only)
        :param callback: callback function to be used inside the method, if applicable
        :return: the respective result value: `Qobj` for `run`, `Counter` for `sample`,
            and numeric type (`float`, `complex`, `ArrayLike`) for `expectation`
        """

        match run_type:
            case "run":
                return platform.get_final_state()
            case "sample":
                return platform.sample_final_state(shots)
            case "expectation":
                return platform.expect(obs_list=observable)
            case _:
                raise NotImplementedError(f"Run type '{run_type}' not implemented.")

    def _on_emulator(
        self,
        run_type: Literal["run", "sample", "expectation"],
        values: Optional[dict[str, float]],
        shots: Optional[int] = None,
        observable: Optional[Any] = None,
        callback: Optional[Callable] = None,
        **_: Any,
    ) -> Any:
        """
        Method that runs over Pulser emulator.


        **Notice**: for now, only `QutipEmulator` is supported.

        It uses `run_type` to decide which kind of run it will perform: `run`, `sample`,
        or `expectation`.

        :param run_type: str: `run`, `sample`, `expectation` possible values
        :param values: dictionary of user-input parameters
        :param shots: int: number of shots; applied only for `sample` option
        :param observable: list of observables; applied only for `expectation` option
        :param callback: callback function to be used inside the method (if applicable)
        :return: the respective result value: `Qobj` for `run`, `Counter` for `sample`,
            and numeric type (`float`, `complex`, `ArrayLike`) for `expectation`
        """
        vals: dict[str, float] = {**(values or {}), **self._params}
        built_sequence: Sequence = self.sequence.build(**vals)  # type: ignore
        simulation: QutipEmulator = QutipEmulator.from_sequence(
            built_sequence, with_modulation=True
        )
        result: SimulationResults = simulation.run()

        return self._run(
            run_type=run_type,
            platform=result,
            shots=shots,
            observable=observable,
            callback=callback,
        )

    def _on_qpu(
        self,
        run_type: Literal["run", "sample", "expectation"],
        values: Optional[dict[str, float]],
        shots: Optional[int] = None,
        observable: Optional[Any] = None,
        callback: Optional[Callable] = None,
        **_: Any,
    ) -> Any:
        """
        Method that runs over QPU.

        **Notice**: for now, it is not implemented yet.

        :param run_type:
        :param values:
        :param shots:
        :param observable:
        :param callback:
        :param _:
        :return:
        """
        raise NotImplementedError("QPU execution not implemented yet.")

    def run(
        self,
        *,
        values: Optional[dict[str, float]] = None,
        on: Literal["emulator", "qpu"] = "emulator",
        callback: Optional[Callable] = None,
        **_: Any,
    ) -> RunResult:
        match on:
            case "emulator":
                return self._on_emulator(
                    run_type="run", values=values, callback=callback
                )
            case "qpu":
                return self._on_qpu(run_type="run", values=values, callback=callback)
            case _:
                raise NotImplementedError(f"Platform '{on}' not implemented.")

    def sample(
        self,
        *,
        values: Optional[dict[str, float]] = None,
        shots: Optional[int] = None,
        on: Literal["emulator", "qpu"] = "emulator",
        callback: Optional[Callable] = None,
        **_: Any,
    ) -> Counter:
        match on:
            case "emulator":
                return self._on_emulator(
                    run_type="sample", values=values, shots=shots, callback=callback
                )
            case "qpu":
                return self._on_qpu(
                    run_type="sample", values=values, shots=shots, callback=callback
                )
            case _:
                raise NotImplementedError(f"Platform '{on}' not implemented.")

    def expectation(
        self,
        *,
        values: Optional[dict[str, float]] = None,
        on: Literal["emulator", "qpu"] = "emulator",
        observable: Optional[Any] = None,
        callback: Optional[Callable] = None,
        **_: Any,
    ) -> Qobj:
        match on:
            case "emulator":
                return self._on_emulator(
                    run_type="expectation",
                    values=values,
                    observable=observable,
                    callback=callback,
                )
            case "qpu":
                return self._on_qpu(
                    run_type="expectation",
                    values=values,
                    observable=observable,
                    callback=callback,
                )
            case _:
                raise NotImplementedError(f"Platform '{on}' not implemented.")
