from abc import abstractmethod
from typing import Any, List

from simulation.messages import SimulationState
from .base import BaseCallable
from .modifier import Modifier, float64_array
from gym import spaces

import numpy

from . import calculations


BOX_LOW: float = -5
BOX_HIGH: float = 5
DEFAULT_INVALID: float = -2


class Observation(BaseCallable):
    def __init__(
        self, low: float, high: float, modifier: Modifier = None, **kwargs
    ) -> None:

        self.low = low
        self.high = high
        self.modifier = modifier
        super().__init__(**kwargs)

    def modify(self, value):
        """Modify the value if the modifier exists"""
        if not self.modifier:
            return value

        return self.modifier(value, self.low, self.high)

    @abstractmethod
    def space(self) -> spaces.Space:
        ...


class DistanceToPoint(Observation):
    def __init__(
        self,
        low: float = 0,
        high: float = 30,
        modifier: Modifier = None,
        **kwargs,
    ) -> None:
        super().__init__(low, high, modifier, **kwargs)

    def space(self) -> spaces.Box:

        return spaces.Box(
            low=numpy.array([BOX_LOW]),
            high=numpy.array([BOX_HIGH]),
            dtype=numpy.float64,
        )

    def __call__(
        self, name: str, other: str, state: SimulationState, **kwargs
    ) -> numpy.array:

        if name is None:
            return float64_array(DEFAULT_INVALID)

        distance = calculations.distance_between(
            state.objects.get(name), state.objects.get(other)
        )

        return float64_array(self.modifier(distance, self.low, self.high))


class AbsoluteBearing(Observation):
    def __init__(
        self,
        low: float = -1,
        high: float = 1,
        modifier: Modifier = None,
        **kwargs,
    ) -> None:
        super().__init__(low, high, modifier, **kwargs)

    def space(self) -> spaces.Box:

        return spaces.Box(
            low=numpy.array([BOX_LOW] * 2),
            high=numpy.array([BOX_HIGH] * 2),
            dtype=numpy.float64,
        )

    def __call__(
        self, name: str, other: str, state: SimulationState, **kwargs
    ) -> Any:

        if name is None:
            return numpy.array([DEFAULT_INVALID] * 2, numpy.float64)

        abs_bearing: float = calculations.absolute_bearing_between(
            state.objects.get(name), state.objects.get(other)
        )

        sin = numpy.sin(abs_bearing)
        cos = numpy.cos(abs_bearing)

        return numpy.array([cos, sin], numpy.float64)


class Speed(Observation):
    def __init__(
        self,
        low: float = 0,
        high: float = 30,
        modifier: Modifier = None,
        **kwargs,
    ) -> None:
        super().__init__(low, high, modifier, **kwargs)

    def space(self) -> spaces.Box:

        return spaces.Box(
            low=numpy.array([BOX_LOW]),
            high=numpy.array([BOX_HIGH]),
            dtype=numpy.float64,
        )

    def __call__(self, name: str, state: SimulationState, **kwargs) -> Any:

        if name is None:
            return float64_array(DEFAULT_INVALID)

        speed: float = state.objects.get(name).speed

        return float64_array(self.modify(speed))


class ValidActions(Observation):
    def __init__(self, fixed: int, variable: int, **kwargs) -> None:

        self.fixed_actions = fixed
        self.variable_actions = variable

    def space(self) -> spaces.Space:
        return spaces.Box(
            low=numpy.array(
                [0] * (self.fixed_actions + self.variable_actions)
            ),
            high=numpy.array(
                [1] * (self.fixed_actions + self.variable_actions)
            ),
            dtype=numpy.float64,
        )

    def __call__(self, variable_list: List, **kwargs) -> Any:

        output = [1] * self.fixed_actions

        assert (
            len(variable_list) == self.variable_actions
        ), f"{len(variable_list)}, {self.variable_actions}"

        output.extend([1 if i is not None else 0 for i in variable_list])

        return numpy.array(output, dtype=numpy.float64)
