# Example

Below you will find a working example on how to use `Qadecen 2 platforms` package. We first need a `Model` data to work with (from [`Qadence 2 IR`](https://github.com/pasqal-io/qadence2-ir)). Once the model is defined, a backend must be chosen. A `compile_to_backend` should be invoked to translate the model data into backend-specific data, and also to expose backend's methods and functionalities to execute the model data code. The execution can be done in emulators or QPU, with sampling or expectation values, for instance.

In the case below, `torch` data is used, so `autograd` can be done in the user input tensor. `PyQTorch` is defined as backend and thus the `Model` will be converted into `torch` tensor data, for a wavefunction calculation through the backend `Interface` method called `run`.


```python exec="on" source="material-block" result="json" session="compile_to_backend"
import torch
import pyqtorch as pyq
from qadence2_ir.types import (
    Model,
    Alloc,
    AllocQubits,
    Call,
    Load,
    Support,
    QuInstruct,
    Assign
)
from qadence2_platforms.compiler import compile_to_backend


# define the model
model = Model(
    register=AllocQubits(num_qubits=2, options={"initial_state": "10"}),
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

# place the model and choose the backend in `compile_to_backend` function
compiled_model = compile_to_backend(model, "pyqtorch")

# define the feature parameters values that are used in the `Model`
feature_params = {"x": torch.rand(1, requires_grad=True)}

# run the `Model` execution to retrieve a wavefunction as `torch.Tensor`
wavefunction = compiled_model.run(state=pyq.zero_state(2), values=feature_params)

# calculate the `grad` from the wavefunction, given the feature parameters
dfdx = torch.autograd.grad(wavefunction, feature_params["x"], torch.ones_like(wavefunction))

print(dfdx)
```
