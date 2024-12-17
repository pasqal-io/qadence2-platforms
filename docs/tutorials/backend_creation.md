# Building a custom backend

It is possible to build your own custom backend on Qadence 2 platforms.

## Why building your own backend

- Define a new device and have a full custom experience
- Customize primitive functions to test or extend ideas, new parameters, etc.
- Customize emulation backend

## How to build your own backend

Here is a brief description on how you can create your own backend:

```python
from pathlib import Path
from qadence2_platforms.utils import BackendTemplate

my_path = Path("../contents")

template = BackendTemplate()
template.create_template("my_backend1", gui=False, use_this_dir=my_path)
```

And it is done! Now you have go through the newly created `custom_backends/my_backend1` folder at the current folder (defined in `Path(".")`) and implement all the necessary methods. `BackendTemplate` instance also creates a few necessary files, such as `compiler.py` and `interface.py` with pre-filled code. There is a comprehensive list of TODOs inside those files so you can properly implement what is needed for your backend to successfully be used by `qadence2-platforms`.
