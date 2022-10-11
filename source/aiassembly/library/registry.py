"""Module for loading implementation-specific functions and definitions"""

from typing import Callable, Optional
from aiassembly.environments.types import FunctionRegistry

REGISTRY: FunctionRegistry = {}


def define_name(name: str, owner: str) -> str:
    """Defines the name of the registered item by its unique name and owner"""
    return f"{name}_{owner}"


def register_function(
    name: str, owner: str, func: Callable, registry: FunctionRegistry = None
) -> None:
    """Registers the function under the given name and owner"""
    if registry is None:
        registry = REGISTRY

    name = define_name(name, owner)

    registry[name] = func


def get_registered_function(
    name: str, owner: str, registry: FunctionRegistry=None
) -> Callable:
    """Returns the function registered by name and owner"""
    if registry is None:
        registry = REGISTRY

    name = define_name(name, owner)
    
    return registry[name]
