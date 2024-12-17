# Qadence 2 Platforms


!!! note
    Qadence 2 Platforms is currently a *work in progress* and is under active development.

    Please be aware that the software is in an early stage, and frequent updates, including breaking changes, are to be expected. This means that:
    * Features and functionalities may change without prior notice.
    * The codebase is still evolving, and parts of the software may not function as intended.
    * Documentation and user guides may be incomplete or subject to significant changes.


Qadence 2 Platforms is a collection of functionalities that transforms [Qadence IR](https://github.com/pasqal-io/qadence2-ir/) into backend-specific data and constructors, to be executed by backend methods. It is not intended to be used directly by [Qadence 2](https://github.com/pasqal-io/qadence2-core/) users, but rather those who need to implement or extend backends, quantum instruction primitives, compiler or backend directives, etc.


## Installation

!!! note
    it is advised to set up a python environment before installing the package.

To install the current version, there is currently one option:


### Installation from Source

Clone this repository by typing on the terminal

```bash
git clone https://github.com/pasqal-io/qadence2-platforms.git
```

Go to `qadence2-platforms` folder and install it using [hatch](https://hatch.pypa.io/latest/):

```bash
python -m pip install hatch
```

and run `hatch` to create or reuse the project environment:

```bash
hatch -v shell
```

## Description

### Platforms

This package **should not** be used directly by the user. It is used to convert [Qadence IR](https://github.com/pasqal-io/qadence2-ir) into backend-compatible data, and to execute it with extra options (provided by the compilation process, either on [Qadence 2 expressions](https://github.com/pasqal-io/qadence2-expressions) or [Qadence 2 core](https://github.com/pasqal-io/qadence2-core)).

### Qadence 2 Intermediate Representation (IR)

Qadence 2 expressions is being compiled into an IR comprised of both quantum and classical operations.

### Platforms API

The `backend` module exposes a single `compile_to_backend` function which accepts a `Model` and a string denoting the `backend`.

### Platforms Backend

Each submodule under `backend` is expected (1) to translate the `IR` data into backend-compatible data, (2) to provide instruction conversions from `IR` to backend, (3) to handle the storage and embedding of parameters, and (4) to implement execution process for `run`, `sample` and `expectation`.

### Usage

Example
```python exec="on" source="material-block" session="model"
from qadence2_ir.types import (
    Model,
    Alloc,
    AllocQubits,
    Call,
    Assign,
    QuInstruct,
    Support,
    Load
)


Model(
    register = AllocQubits(
        num_qubits = 3,
        qubit_positions = [(-2,1), (0,1), (1,3)],
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
		        Support(control=(0,), target=(2,)),
		        Load("t"),
		        Load("Omega"),
		        Load("delta"),
		    )
    ],
    directives = {"digital-analog": True},
)
```

Compiling a `pyqtorch` circuit and computing gradients using `torch.autograd`

```python exec="on" source="material-block" session="model"
import torch
import pyqtorch as pyq
from qadence2_ir.types import (
    Model, Alloc, AllocQubits, Load, Call, Support, QuInstruct, Assign
)

from qadence2_platforms.compiler import compile_to_backend


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
)
api = compile_to_backend(model, "pyqtorch")
f_params = {"x": torch.rand(1, requires_grad=True)}
wf = api.run(state=pyq.zero_state(2), values=f_params)
dfdx = torch.autograd.grad(wf, f_params["x"], torch.ones_like(wf))[0]
```

## Documentation

!!! note
    Documentation in progress.


## Contribute

Before making a contribution, please review our [code of conduct](docs/getting_started/CODE_OF_CONDUCT.md).

- **Submitting Issues:** To submit bug reports or feature requests, please use our [issue tracker](https://github.com/pasqal-io/qadence2-platforms/issues).
- **Developing in qadence 2 platforms:** To learn more about how to develop within `qadence 2 platforms`, please refer to [contributing guidelines](docs/getting_started/CONTRIBUTING.md).
