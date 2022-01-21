from dataclasses import dataclass, field
from typing import List
import simulation


@dataclass
class Event:

    names: set = field(default_factory=set)

    def add(self, name: str):

        self.names.add(name)


@dataclass
class CollisionEvent(Event):
    ...


@dataclass
class TaskCompletedEvent(Event):
    ...


@dataclass
class NewParticles(Event):
    ...


def check_for_collisions(particles: dict):

    names: set = set(particles.keys())
    collisions: List[CollisionEvent] = []

    while len(names) > 1:

        to_check: str = names.pop()

        for other_name in names:
            # need to check if in either's radius
            in_bounds = simulation.calculations.in_bounds_of(
                particles[to_check], particles[other_name]
            ) or simulation.calculations.in_bounds_of(
                particles[other_name], particles[to_check]
            )

            if not in_bounds:
                continue

            event = CollisionEvent({to_check, other_name})
            collisions.append(event)

    return collisions
