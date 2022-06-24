from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List
from easyai.simulation.scripts import Script
from easyai.simulation.messages import Message, SimulationState
from .particles import (
    Particle,
)
import pprint


@dataclass
class Simulator:
    """
    Simulation that maintains and updates
    particles on a fixed timestep.

    Supports ability to add fixed scripts
    to manipulate objects, as well
    as a general interface to register listeners
    for queries
    """

    step_size: float
    time: float = 0
    objects: dict = field(default_factory=dict)
    scripts: List = field(default_factory=list)

    listeners: list = field(default_factory=list)

    def __str__(self):
        return pprint.pformat(asdict(self))

    def add_particle(self, particle: Particle) -> None:
        """Adds a particle to the simulation.

        :param particle: particle object
        :type particle: Particle
        :raises KeyError: If name exists
        """

        if particle.name in self.objects:
            raise KeyError(f"{particle.name} already exists!")

        self.objects[particle.name] = particle
        particle.simulator = self

    def remove_particle(self, name: str) -> None:
        """Removes the provided name from the sim"""
        self.objects.pop(name)

    def remove_all_particles(self) -> None:
        """
        Remove all the objects from the sim,
        """

        self.objects.clear()

    def remove_all_scripts(self) -> None:

        self.scripts.clear()

    def remove_all(self):

        self.remove_all_particles()
        self.remove_all_scripts()

    def reset(self):

        self.time = 0
        for listener in self.listeners:

            listener.notify(SimulationState(self.time, self.objects))

    def get(self, name: str) -> Particle:
        """Returns the particle of the given name"""

        return self.objects[name]

    def register_script(self, script: Script):
        """Adds a callable script to the simulation
        to call and maintain

        :param script: [description]
        :type script: Script
        """
        self.scripts.append(script)

    def register_listener(self, listener):
        """Registers a listener

        :param listener: _description_
        :type listener: _type_
        """

        self.listeners.append(listener)

    def notify_listeners(self, event: Message):
        """Notifies the listeners of the message

        :param event: _description_
        :type event: Message
        """
        for listener in self.listeners:

            listener.notify(event)

    def update(self) -> None:
        """Updates the objects given the delta timestep in seconds"""

        for obj in self.objects.values():

            obj.update(timestep=self.step_size)

        for script in self.scripts:
            script.update(timestep=self.step_size, simulator=self)

        self.time += self.step_size

        for listener in self.listeners:

            listener.notify(SimulationState(self.time, self.objects))
