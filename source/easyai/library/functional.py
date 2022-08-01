"""Module for building generic functional patterns"""

import functools
from typing import Any, Callable, List


def build_functional_chain(funcs: List[Callable]) -> Callable[[Any], Any]:
    """
    Build a definition of functions where the result of previous is
    passed to next func
    """
    return functools.partial(
        functools.reduce, function=lambda x, y: x(y), sequence=funcs
    )


def functional_extraction(obj: Any, funcs: List[Callable]) -> List[Any]:
    """Pass a single object to a series of functions and return their output"""
    return list(map(lambda x: x(obj), funcs))


def build_functional_extraction(funcs: List[Callable]) -> List[Any]:
    """Wraps functional_extraction as a partial"""
    return functools.partial(functional_extraction, funcs=funcs)
