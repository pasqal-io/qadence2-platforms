# Backends

The backends are a collection of platform-specific functionalities to provide the proper IR model data transformation and execution on chosen simulators or real devices. They are designed to behave in a standardized way to provide code extensibility and are flexible enough so advanced users can [create their own backends](backend_creation.md).

Current available built-in backends are [`Fresnel-1`](../api/backends/fresnel1/index.md), [`AnalogDevice`](../api/backends/analog/index.md), and [`PyQTorch`](../api/backends/pyqtorch/index.md).
