from dataclasses import dataclass, field
from typing import List
from simulation.rendering import Render, RenderData
from simulation.scripts import Script
from simulation.events import check_for_collisions, TaskCompletedEvent

from .particles import (
    Particle,
    Types,
    create_position,
)


@dataclass
class SimpleWorld:
    """
    Simulation that maintains and updates
    particles on a fixed timestep.

    Supports ability to add fixed scripts
    to manipulate objects, as well
    as a general interface for simple queries
    """

    step_size: float
    time: float = 0
    objects: dict = field(default_factory=dict)
    type_counter: dict = field(default_factory=dict)
    scripts: List = field(default_factory=list)

    _added_particles: List = field(default_factory=list)
    _rendering: Render = None

    def add_render(self, render: Render):
        """Add a renderer to the sim"""
        self._rendering = render

    def add_object(self, particle: Particle) -> None:
        """Adds a particle to the simulation.

        :param particle: particle object
        :type particle: Particle
        :raises KeyError: If name exists
        """

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
        """
        Remove all the objects from the sim,
        resets any other containers to
        empty state.
        """

        self.objects.clear()
        self.scripts.clear()
        self.type_counter.clear()
        self._added_particles.clear()
        self.time = 0

    def get_all_of_type(self, type: Types):
        """Returns particles of matching type

        :param type: [description]
        :type type: Types
        :return: [description]
        :rtype: [type]
        """
        return tuple(self.type_counter.get(type.name, ()))

    def get(self, name: str) -> Particle:
        """Returns the particle of the given name"""

        return self.objects[name]

    def create_particle(self, name: str, type: Types) -> Particle:
        """Creates a particle of given unique name and built
        in type.

        :param name: [description]
        :type name: str
        :param type: [description]
        :type type: Types
        :return: [description]
        :rtype: Particle
        """
        part = Particle(
            name=name, position=create_position(0, 0, 0), type=type
        )

        self.add_object(part)

        return part

    def add_script(self, script: Script):
        """Adds a callable script to the simulation
        to call and maintain

        :param script: [description]
        :type script: Script
        """
        self.scripts.append(script)

    def get_collision_events(self) -> List:
        """Query for collisions"""
        return check_for_collisions(self.objects)

    def get_untasked_agents(self) -> List:
        """Return list of particles with no task

        :return: [description]
        :rtype: List
        """
        output = []
        for particle in self.objects.values():

            if particle.tasked:
                continue

            event = TaskCompletedEvent()
            event.add(particle.name)
            output.append(event)

        return output

    def get_added_particles(self) -> List:
        """Get particles added in last
        timestep

        :return: [description]
        :rtype: List
        """
        return list(self._added_particles)

    def update(self) -> None:
        """Updates the objects given the delta timestep in seconds"""

        # any new particles added in this
        # timestep would now be added to new list
        self._added_particles.clear()

        for obj in self.objects.values():

            obj.update(timestep=self.step_size)

        for script in self.scripts:
            script.update(timestep=self.step_size, simulator=self)

        self.time += self.step_size

        if self._rendering is not None:
            render_data = {
                part.name: RenderData(
                    part.name, part.position[0], part.position[1], part.radius
                )
                for part in self.objects.values()
            }
            self._rendering.render(render_data)
