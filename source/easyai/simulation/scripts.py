from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
import typing
from easyai.simulation.particles import create_particle
from easyai.simulation.types import Types

if typing.TYPE_CHECKING:
    from easyai.simulation.simulator import Simulator
    from easyai.simulation.particles import Particle


class Script(ABC):
    @abstractmethod
    def update(self, timestep: float, **kwargs) -> None:
        ...


@dataclass
class CreateEntityInterval(Script):
    prefix: str
    type: Types
    interval: float
    setup_func: Callable = None
    max: int = 1e3
    _last_created: float = 0
    _created: int = 0

    def get_of_type(self, simulator: Simulator, type: Particle):

        return [
            part for part in simulator.objects.values() if part.type == type
        ]

    def update(self, simulator, timestep: float, **kwargs) -> None:

        if self._last_created + timestep < self.interval:
            self._last_created += timestep
            return

        if len(self.get_of_type(simulator, self.type)) >= self.max:
            return

        self._last_created = 0

        particle = create_particle(
            name=f"{self.prefix}-{self._created}", type=self.type
        )

        if self.setup_func:
            self.setup_func(particle)

        simulator.add_particle(particle)

        self._created += 1
