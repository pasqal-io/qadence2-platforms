# Qadence 2 Platforms
Platform dependent APIs and engines (backends) to be used on Qadence 2.


## Installation
Installation guidelines

## Usage

Example
```python
Model(
    Register(
        type = "triangular",
        scale = 1.0,
        allocate = [(-2,1), (0,1), (1,3)],
        options = {"initial_state": "010"}
    ),
    inputs = {
        "x": Alloc(1, trainable=False),
        "t": Alloc(1, trainable=False),      # time
        "Omega": Alloc(4, trainable=True),   # 4-points amp. modulation
        "delta": Alloc(1, trainable=False)), # detuning
    },
    instructions = [
        # -- Feature map
        Assign("%0", Call("mul", 1.57, Load("x")),
        Assign("%1", Call("sin", Load("%0"))),
        QuInstruct("rx", Support(target=(0,)), Load("%1")),
        # --
        QuInstruct("h", Support.target_all()),
        QuInstruct("not", Support(target=(1,), control=(0,))),
        QuInstruct(
		        "qubit_dyn",
		        Support(0, 2),
		        Load("t"),
		        Load("Omega"),
		        Load("delta"),
		    )
    ],
    directives = {"digital-analog": True},
    backend_options = {"result-type": "state-vector", "data-type": "f32"}
)
```

## Documentation
Documentation guidelines

## Contribute
Contribution guidelines
