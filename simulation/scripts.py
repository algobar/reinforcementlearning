from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
from simulation.objects import Types


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

    def update(self, simulator, timestep: float, **kwargs) -> None:

        if self._last_created + timestep < self.interval:
            self._last_created += timestep
            return

        if len(simulator.get_all_of_type(self.type)) >= self.max:
            return

        self._last_created = 0

        particle = simulator.create_particle(
            name=f"{self.prefix}-{self._created}", type=self.type
        )

        if self.setup_func:
            self.setup_func(particle)

        self._created += 1
