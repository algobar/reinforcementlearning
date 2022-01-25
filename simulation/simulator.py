from dataclasses import dataclass, field
from typing import List
from simulation.scripts import Script
from .particles import (
    Particle,
    Types,
    create_position,
)

from simulation.events import check_for_collisions, TaskCompletedEvent


@dataclass
class SimpleWorld:

    step_size: float
    time: float = 0
    objects: dict = field(default_factory=dict)
    type_counter: dict = field(default_factory=dict)
    scripts: List = field(default_factory=list)

    _added_particles: List = field(default_factory=list)

    def add_object(self, particle: Particle) -> None:

        if particle.name in self.objects:
            raise KeyError(f"{particle.name} already exists!")

        # keep record of type
        if particle.type.name not in self.type_counter:
            self.type_counter[particle.type.name] = set()

        self.type_counter[particle.type.name].add(particle.name)

        self.objects[particle.name] = particle

        self._added_particles.append(particle.name)

    def remove_object(self, name: str) -> None:
        """Removes the provided name from the sim"""
        particle: Particle = self.objects.pop(name)
        self.type_counter[particle.type.name].remove(name)

    def reset(self) -> None:
        """Remove all the objects from the sim"""

        self.objects.clear()
        self.scripts.clear()
        self.type_counter.clear()
        self._added_particles.clear()
        self.time = 0

    def get_all_of_type(self, type: Types):

        return tuple(self.type_counter.get(type.name, ()))

    def get(self, name: str) -> Particle:

        return self.objects[name]

    def create_particle(self, name: str, type: Types) -> Particle:

        part = Particle(name=name, position=create_position(0, 0, 0), type=type)

        self.add_object(part)

        return part

    def add_script(self, script: Script):

        self.scripts.append(script)

    def get_collision_events(self) -> List:

        return check_for_collisions(self.objects)

    def get_untasked_agents(self) -> List:

        output = []
        for particle in self.objects.values():

            if particle.tasked:
                continue

            event = TaskCompletedEvent()
            event.add(particle.name)
            output.append(event)

        return output

    def get_added_particles(self) -> List:

        return list(self._added_particles)

    def update(self) -> None:
        """Updates the objects given the delta timestep in seconds"""

        # any new particles added in this
        # timestep would now be added to new list
        self._added_particles.clear()

        for obj_name in self.objects:

            self.objects[obj_name].update(timestep=self.step_size)

        for script in self.scripts:
            script.update(timestep=self.step_size, simulator=self)

        self.time += self.step_size
