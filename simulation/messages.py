"""Module to define messages used for simulation state, events, etc"""


from dataclasses import dataclass, field
from typing import Dict, List, Protocol

from simulation.particles import Particle


class Message(Protocol):
    """Used for type hinting. No methods at this time but may be expanded"""

    ...


@dataclass
class SimulationState:
    """Contains all the objects current in the simulation, as well as time"""

    time: float
    objects: Dict[str, Particle] = field(default_factory=dict)


@dataclass
class ParticleAdded:
    """Contains a particle added to the scenario"""

    time: float
    particle: Particle


@dataclass
class TaskComplete:
    """Indicates a particle that has had a task completed"""

    time: float
    particle: Particle


@dataclass
class Collision:
    """Indicates two particles that have collided by entering both or the other particle's zone"""

    time: float
    particles: List[Particle] = field(default_factory=list)
