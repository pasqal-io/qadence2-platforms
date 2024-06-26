# Qadence 2 Platforms
Platform dependent APIs and engines (backends) to be used on Qadence 2.


## Installation
Installation guidelines

## Qadence Intermediate Representation
Qadence2 expressions is being compiled into an IR comprised of both quantum and classical operations.
## API
The `backend` module exposes a single `compile` function which accepts a `Model` and a string denoting the `backend`.
## Backend
Each backend needs to handle the storage and embedding of parameters in a `Embedding` class
and the compilation of `model.instructions` into native instructions in the particular backend via `compile`.
Each backend is expected to expose a `Embedding` class and a `compile` function.

## Usage

Example
```python
from qadence2_platforms import Model

Model(
    register = AllocQubits(
        num_qubits = 3,
        positions = [(-2,1), (0,1), (1,3)],
        grid_type = "triangular",
        grid_scale = 1.0,
        options = {"initial_state": "010"}
    ),
    inputs = {
        "x": Alloc(1, trainable=False),
        "t": Alloc(1, trainable=False),      # time
        "Omega": Alloc(4, trainable=True),   # 4-points amp. modulation
        "delta": Alloc(1, trainable=False), # detuning
    },
    instructions = [
        # -- Feature map
        Assign("%0", Call("mul", 1.57, Load("x"))),
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
    data_settings = {"result-type": "state-vector", "data-type": "f32"}
)
```

Compiling a `pyqtorch` circuit and computing gradients using `torch.autograd`

```python
model = Model(
    register=AllocQubits(num_qubits=2),
    inputs={
        "x": Alloc(size=1, trainable=False),
    },
    instructions=[
        Assign("%0", Call("mul", 1.57, Load("x"))),
        Assign("%1", Call("sin", Load("%0"))),
        QuInstruct("rx", Support(target=(0,)), Load("%1")),
        QuInstruct("not", Support(target=(1,), control=(0,))),
    ],
    directives={"digital": True},
    data_settings={"result-type": "state-vector", "data-type": "f64"},
)
api = compile(model, "pyqtorch")
f_params = {"x": torch.rand(1, requires_grad=True)}
wf = api.run(pyq.zero_state(2), f_params)
dfdx = torch.autograd.grad(wf, f_params["x"], torch.ones_like(wf))[0]
```

## Documentation
Documentation guidelines

## Contribute
Contribution guidelines
