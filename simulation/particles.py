from dataclasses import dataclass
from typing import Callable
from enum import Enum, auto
import numpy

from simulation.simulator import Simulator
from simulation.messages import TaskComplete


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
    speed: float = 0
    radius: float = 0

    _simulator: Simulator = None
    _behavior: Callable = None
    _tasked: bool = False

    @property
    def tasked(self):

        return self._tasked

    @tasked.setter
    def tasked(self, value: bool):

        # setting task to false will trigger
        # a complete
        if not value:
            self.register_task_complete()

        self._tasked = value

    @property
    def behavior(self):

        return self._behavior

    @behavior.setter
    def behavior(self, behavior):

        self._behavior = behavior

        if self._behavior is None:
            self.tasked = False

        self.tasked = True

    @property
    def simulator(self) -> Simulator:

        return self._simulator

    @simulator.setter
    def simulator(self, simulator: Simulator):

        # setting the simulator
        # registers the particle
        self._simulator = simulator

    def register_task_complete(self) -> None:

        self.simulator.notify_listeners(
            TaskComplete(self.simulator.time, self)
        )

    def set_position(self, x: float, y: float, z: float) -> None:

        self.position[0] = x
        self.position[1] = y
        self.position[2] = z

    def update_behavior(self, **kwargs) -> None:

        if self.behavior is None:
            return

        if self.behavior(particle=self, **kwargs):
            self.behavior = None

    def update(self, **kwargs) -> None:
        """Updates the object, replaces the behavior if a new one is provided"""

        self.update_behavior(**kwargs)


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


def create_particle(name: str, position: numpy.array, type: Types):

    return Particle(name, position, type)
