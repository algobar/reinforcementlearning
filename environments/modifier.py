from .base import BaseCallable
from typing import Any
import numpy


def float64_array(value):
    """Creates a float 32 array"""
    return numpy.array([value], dtype=numpy.float64)


class Modifier(BaseCallable):
    """Modify observations"""

    ...


class NormalizeBox(Modifier):
    @staticmethod
    def normalize_box(
        value: float,
        low: float,
        high: float,
        norm_low: float = -1.0,
        norm_high: float = 1.0,
    ):
        """Bound the given value given a box between specified low and high"""
        return numpy.interp(
            value,
            [low, high],
            [norm_low, norm_high],
        )

    def __call__(self, value: float, low: float, high: float, **kwargs) -> Any:
        return self.normalize_box(value, low, high)
