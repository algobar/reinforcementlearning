from abc import ABC, abstractmethod
from typing import Callable
import numpy
from dataclasses import dataclass, field
from functools import partial
from simulation.calculations import magnitude, straight_line_path_2d
from simulation.objects import Particle


class Behavior(ABC):
    @abstractmethod
    def __call__(self, particle: Particle, timestep: float, **kwds) -> bool:
        ...


@dataclass
class GoToPoint2D(Behavior):

    end: numpy.array
    speed: float
    threshold: float = 0.1
    _behavior: Callable = field(init=False)

    def __post_init__(self):

        self.end = numpy.copy(self.end)
        self._behavior = partial(
            straight_line_path_2d, end=self.end, speed=self.speed
        )

    def __call__(self, particle: Particle, timestep: float, **kwds) -> bool:

        particle.position = self._behavior(
            current=particle.position, timestep=timestep
        )

        return magnitude(particle.position - self.end) < self.threshold


@dataclass
class RemainInLocationSeconds:
    total_time: float
    _time_waited: float = 0.0

    def __call__(self, timestep: float, **kwds) -> bool:

        self._time_waited += timestep

        return self._time_waited >= self.total_time
