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
        Instruction("rx", Support(0), Parameter("x0", 1, trainable=False)),
        Instruction("rx", Support(1), 5.2),
        Instruction("h", Support.all),
        Instruction(
            "qubit_dym",
            Support(0, 2),
            Paramter("t", 1, trainable=False),  # time
            Paramter("Omega", 4, trainable=True),  # Amplitude modulation with 4 points
            Paramter("delta", 1, trainable=False),  # detuning
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
