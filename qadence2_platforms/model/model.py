"""
This module contains:
- models class
- register class
- directive/pragma class
- instruction class
"""

from __future__ import annotations

from typing import Any

from qadence2_platforms.generics import InstructionSet


class BackendConfig:
    """
    Configuration class for backend-specific options. For instance,
    defining the data type, such as `int64`, `bound-dimension` on emu-t.
    """

    def __init__(self, **kwargs: Any):
        pass


class Register:
    """
    Register class to define backend-specific register data structures. For
    instance, grid scale, atom coordinates, etc.
    """

    pass


class Directives:
    """
    Directive class defines backend-independent options during compilation
    process. For instance, whether the backend accepts digital or analog
    blocks, rydberg level, automatic differentiation, etc.
    """

    pass


class Models:
    """
    Models class aggregates all the data above in a nice packed way. It then
    is used on the backend that will define how to deal with data.

    Backend needs to check the content of `Models` attributes and data, and
    then starts transpiling into backend-specific data structure.
    """

    def __init__(
        self,
        directives: Directives,
        register: Register,
        instructions: InstructionSet,
        config: BackendConfig,
    ):
        self.directives = directives
        self.register = register
        self.instr = instructions
        self.config = config
