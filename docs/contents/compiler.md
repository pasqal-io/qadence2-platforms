# Compiling to backend

The Qadence 2 internals rely on a few function calls to go from expressions to backend execution. On Qadence 2 platforms, the function responsible for this is [`compile_to_backend`](../api/compiler.md). It creates the bridge between Qadence 2 IR model data and the correct backend defined by the user (or by the compiler). When the user chooses the backend name on Qadence 2 ([`qadence2-core`](https://github.com/pasqal-io/qadence2-core)) at the [`code_compile`](https://github.com/pasqal-io/qadence2-core/blob/main/qadence2_core/compiler.py) function, this information is passed to `compile_to_backend` to look for the correct backend (using [`module_importer`](../api/utils/module_importer.md) logic) and, if it exists, backend's `compile_to_backend` function will be invoked, generating an `Interface` instance.