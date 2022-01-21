from dataclasses import dataclass, field
from typing import Callable, List
from enum import Enum, auto
import numpy


def create_position(x: float, y: float, z: float) -> numpy.array:
    """Creates a new position given the initial coordinates

    Args:
        x (float): [description]
        y (float): [description]
        z (float): [description]

    Returns:
        numpy.array: represents coordinates in 3 dimensions
    """
    return numpy.array([x, y, z], dtype=numpy.float32)


class Types(Enum):
    """Represents types of objects in the simulation"""

    AGENT = auto()
    BASE = auto()
    ENEMY = auto()


@dataclass
class Particle:
    """Represents a simple object with position and speed"""

    name: str
    position: numpy.array
    type: Types
    radius: float = 0
    speed: float = 0
    behavior: Callable = None
    tasked: bool = False

    def set_position(self, x: float, y: float, z: float) -> None:

        self.position[0] = x
        self.position[1] = y
        self.position[2] = z

    def set_radius(self, radius: float):

        self.radius = radius

    def add_behavior(self, behavior, **kwargs):

        self.behavior = behavior
        self.tasked = True

    def remove_behavior(self):

        self.behavior = None
        self.tasked = False

    def update_behavior(self, **kwargs) -> None:

        if self.behavior is None:
            return

        if self.behavior(particle=self, **kwargs):
            self.remove_behavior()

    def update(self, **kwargs) -> None:
        """Updates the object, replaces the behavior if a new one is provided"""

        self.update_behavior(**kwargs)
