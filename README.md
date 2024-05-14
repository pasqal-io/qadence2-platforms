# Qadence 2 Platforms
Platform dependent APIs and engines (backends) to be used on Qadence 2.


## Installation
Installation guidelines

## Usage

Example
```python
Model(
    Register(
        qubits_positions = [(-2,1), (0,1), (1,3)],
        grid_type = "triangular",
        grid_scale = 1.0,
        options = {"initial_state": "010"}
    ),
    [
        Instruction("rx", (0,), Parameter("x0")),
        Instruction("rx", (1,), Parameter("x1")),
        Instruction("h", ()),  # () == All qubits
        Instruction(
            "qubit_dym",
            (0, 2),
            Paramter("t", 1, mutable=True),  # time
            Paramter("Omega", 4, mutable=False),  # Amplitude modulation with 4 points
            Paramter("delta", 1, mutable=False),  # detuning
        )
    ],
    directives = {"enable_digital_analog": True},
    backend_settings = {"return_type": "state-vector"}
)
```

## Documentation
Documentation guidelines

## Contribute
Contribution guidelines
