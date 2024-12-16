from __future__ import annotations

from collections import Counter
from typing import Any, Union, cast, Callable

from pulser.sequence.sequence import Sequence
from pulser_simulation.simresults import SimulationResults
from pulser_simulation.simulation import QutipEmulator
from qutip import Qobj

from qadence2_platforms import AbstractInterface
from qadence2_platforms.abstracts import OnEnum, RunEnum
from qadence2_platforms.backends._base_analog.functions import base_parse_native_observables
from qadence2_platforms.backends.utils import InputType

RunResult = Union[Counter, Qobj]


class Interface(AbstractInterface[float, Sequence, float, RunResult, Counter, Qobj]):
    def __init__(self, sequence: Sequence, non_trainable_parameters: set[str]) -> None:
        self._non_trainable_parameters = non_trainable_parameters
        self._params: dict[str, float] = dict()
        self._sequence = sequence

    @property
    def info(self) -> dict[str, Any]:
        return {"device": self.sequence.device, "register": self.sequence.register}

    @property
    def sequence(self) -> Sequence:
        return self._sequence

    def parameters(self) -> dict[str, float]:
        return self._params

    def set_parameters(self, params: dict[str, float]) -> None:
        valid_params = params.keys() & self._non_trainable_parameters

        if valid_params != params.keys():
            raise ValueError(f"{set(params.keys())} are not fixed parameters in this sequence.")

        self._params = params

    def draw(self, values: dict[str, Any]) -> None:
        self.sequence.build(**values).draw()

    def _run(
        self,
        run_type: RunEnum,
        platform: SimulationResults,
        shots: int | None = None,
        observable: list[InputType] | InputType | None = None,
        **_: Any,
    ) -> Any:
        """
        Method to be used for `_on_emulator` or `_on_qpu` to execute the respective.

        `run_type` option.

        **Notice**: for now, it only supports `emulator` option and `QutipEmulator`
        platform.

        :param run_type: str: `run`, `sample`, `expectation` options
        :param platform: callable to retrieve methods for executing the options above
        :param shots: number of shots, if applicable (`sample` only)
        :param observable: list of observables, if applicable (`expectation` only)
        :param callback: callback function to be used inside the method, if applicable
        :return: the respective result value: `Qobj` for `run`, `Counter` for `sample`,
            and numeric type (`float`, `complex`, `ArrayLike`) for `expectation`
        """

        match run_type:
            case RunEnum.RUN:
                return platform.get_final_state()
            case RunEnum.SAMPLE:
                return platform.sample_final_state(shots)
            case RunEnum.EXPECTATION:
                if observable is not None:
                    return platform.expect(
                        obs_list=base_parse_native_observables(
                            num_qubits=len(self.sequence.register.qubit_ids), observable=observable
                        )
                    )
                raise ValueError("observable cannot be None or empty on 'expectation' method.")
            case _:
                raise NotImplementedError(f"Run type '{run_type}' not implemented.")

    def _on_emulator(
        self,
        run_type: RunEnum,
        values: dict[str, float] | None,
        shots: int | None = None,
        observable: list[InputType] | InputType | None = None,
        **_: Any,
    ) -> Any:
        """
        Runs on Pulser emulator.

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
        vals: dict[str, float] = {**(values or dict()), **self._params}
        pulse_sequence: Sequence = self.sequence.build(**vals)  # type: ignore
        simulation: QutipEmulator = QutipEmulator.from_sequence(
            pulse_sequence, with_modulation=True
        )
        result: SimulationResults = simulation.run()

        return self._run(
            run_type=run_type,
            platform=result,
            shots=shots,
            observable=observable,
        )

    def _on_qpu(
        self,
        run_type: RunEnum,
        values: dict[str, float] | None,
        shots: int | None = None,
        observable: list[InputType] | InputType | None = None,
        **_: Any,
    ) -> Any:
        """
        Runs on available QPU.

        **Notice**: for now, it is not implemented yet.

        :param run_type: str: `run`, `sample`, `expectation` possible values
        :param values: dictionary of user-input parameters
        :param shots: number of shots
        :param observable: list of observables; applied only for `expectation` option
        :param callback: callback function to be used inside the method (if applicable)
        :return: the respective result value: `Qobj` for `run`, `Counter` for `sample`,
            and numeric type (`float`, `complex`, `ArrayLike`) for `expectation`
        """
        raise NotImplementedError("QPU execution not implemented yet.")

    def run(
        self,
        values: dict[str, float] | None = None,
        on: OnEnum = OnEnum.EMULATOR,
        shots: int | None = None,
        **_: Any,
    ) -> RunResult:
        match on:
            case OnEnum.EMULATOR:
                return self._on_emulator(
                    run_type=RunEnum.RUN,
                    values=values,
                    shots=shots,
                )
            case OnEnum.QPU:
                return self._on_qpu(
                    run_type=RunEnum.RUN,
                    values=values,
                    shots=shots,
                )
            case _:
                raise NotImplementedError(f"Platform '{on}' not implemented.")

    def sample(
        self,
        values: dict[str, float] | None = None,
        shots: int | None = None,
        on: OnEnum = OnEnum.EMULATOR,
        **_: Any,
    ) -> Counter:
        match on:
            case OnEnum.EMULATOR:
                return cast(
                    Counter,
                    self._on_emulator(
                        run_type=RunEnum.SAMPLE,
                        values=values,
                        shots=shots,
                    ),
                )
            case OnEnum.QPU:
                return cast(
                    Counter,
                    self._on_qpu(
                        run_type=RunEnum.SAMPLE,
                        values=values,
                        shots=shots,
                    ),
                )
            case _:
                raise NotImplementedError(f"Platform '{on}' not implemented.")

    def expectation(
        self,
        values: dict[str, float] | None = None,
        observable: list[InputType] | InputType | None = None,
        on: OnEnum = OnEnum.EMULATOR,
        shots: int | None = None,
        **_: Any,
    ) -> Qobj:
        match on:
            case OnEnum.EMULATOR:
                return self._on_emulator(
                    run_type=RunEnum.EXPECTATION,
                    values=values,
                    shots=shots,
                    observable=observable,
                )
            case OnEnum.QPU:
                return self._on_qpu(
                    run_type=RunEnum.EXPECTATION,
                    values=values,
                    shots=shots,
                    observable=observable,
                )
            case _:
                raise NotImplementedError(f"Platform '{on}' not implemented.")
