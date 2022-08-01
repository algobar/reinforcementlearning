"""Module for loading implementation-specific functions and definitions"""

from typing import Callable, Optional
from easyai.environments.types import FunctionRegistry

REGISTRY: FunctionRegistry = {}


def define_name(name: str, owner: str) -> str:

    return f"{name}_{owner}"


def register_function(
    name: str, owner: str, func: Callable, registry: Optional[FunctionRegistry] = None
) -> None:

    if registry is None:
        registry = REGISTRY

    name = define_name(name, owner)

    registry[name] = func


def get_registered_function(
    name: str, owner: str, registry: Optional[FunctionRegistry]
) -> Callable:

    if registry is None:
        registry = REGISTRY

    name = define_name(name, owner)
    
    return registry[name]
