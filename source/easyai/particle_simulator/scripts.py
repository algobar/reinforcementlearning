from typing import Callable
from easyai.environments.types import EntityType, Seconds

def create_entity_interval(
    prefix: str,
    type: EntityType,
    interval: Seconds,
    max: int,
    setup_func: Callable,
):
    ...

# def get_of_type(self, simulator: Simulator, type: Particle):

#     return [
#         part for part in simulator.objects.values() if part.type == type
#     ]

# def update(self, simulator, timestep: float, **kwargs) -> None:

#     if self._last_created + timestep < self.interval:
#         self._last_created += timestep
#         return

#     if len(self.get_of_type(simulator, self.type)) >= self.max:
#         return

#     self._last_created = 0

#     particle = create_particle(
#         name=f"{self.prefix}-{self._created}", type=self.type
#     )

#     if self.setup_func:
#         self.setup_func(particle)

#     simulator.add_particle(particle)

#     self._created += 1
