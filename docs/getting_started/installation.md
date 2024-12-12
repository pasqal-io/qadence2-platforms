# Qadence 2 Platforms


!!! note
    Qadence 2 Platforms is currently a *work in progress* and is under active development.

    Please be aware that the software is in an early stage, and frequent updates, including breaking changes, are to be expected. This means that:

    * Features and functionalities may change without prior notice.
    * The codebase is still evolving, and parts of the software may not function as intended.
    * Documentation and user guides may be incomplete or subject to significant changes.


Qadence 2 Platforms is a collection of functionalities that transforms [Qadence IR](https://github.com/pasqal-io/qadence2-ir/) into backend-specific data and constructors, to be executed by backend methods. It is not intended to be used directly by [Qadence 2](https://github.com/pasqal-io/qadence2-core/) users, but rather only those who need to implement new or extend existing backends, quantum instruction primitives, and compiler or backend directives, etc.


## Installation

!!! note
    It is advised to set up a python environment before installing the package, such as [venv](https://docs.python.org/3/library/venv.html#creating-virtual-environments), [hatch](https://hatch.pypa.io/latest/), [pyenv](https://github.com/pyenv/pyenv), [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [poetry](https://python-poetry.org/). (Qadence 2 in development mode uses `hatch`).

To install the current version, there are a few option:

## Installation from PYPI

On the terminal, type

```bash
pip install qadence2-platforms
```


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

This package **should not** be used directly by the user. It is used to convert [Qadence 2 IR](https://github.com/pasqal-io/qadence2-ir) into backend-compatible data, and to execute it with extra options (provided by the compilation process, either on [Qadence 2 expressions](https://github.com/pasqal-io/qadence2-expressions) or [Qadence 2 core](https://github.com/pasqal-io/qadence2-core)).

### Qadence 2 Intermediate Representation (Q2IR)

Qadence 2 expressions is compiled into an IR comprised of both quantum and classical operations.

### Platforms API

The `backend` module exposes a single `compile_to_backend` function which accepts a `Model` and a string denoting the `backend`.

### Platforms Backend

Each submodule under `backend` is expected (1) to translate the `IR` data into backend-compatible data, (2) to provide instruction conversions from `IR` to backend, (3) to handle the storage and embedding of parameters, and (4) to implement execution process for `run`, `sample` and `expectation`.

## Example

Check the [Example](example.md) tab for usage example of this package.


## Contributing

Before making a contribution, please review our [code of conduct](CODE_OF_CONDUCT.md).

- **Submitting Issues:** To submit bug reports or feature requests, please use our [issue tracker](https://github.com/pasqal-io/qadence2-platforms/issues).
- **Developing in qadence 2 platforms:** To learn more about how to develop within `qadence 2 platforms`, please refer to [contributing guidelines](CONTRIBUTING.md).
