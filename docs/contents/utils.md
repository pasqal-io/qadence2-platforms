# Utils

This module contains functionalities to handle custom backend creation and dynamic importing of backends.

Custom backends can be created through the [`BackendTemplate`](../api/utils/backend_template.md) class which will provide all the necessary code template that must exist in order for Qadence 2 platforms to properly find and generate the `Interface` instance. For more information on how to create a custom backend, check [this tutorial page](../tutorials/backend_creation.md).

The [`module_importer`](../api/utils/module_importer.md) provides a dynamic import for both built-in and custom backends.
