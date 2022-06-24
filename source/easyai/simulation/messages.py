"""Module to define messages used for simulation state, events, etc"""

from __future__ import annotations
import typing
from dataclasses import dataclass, field
from typing import Dict, List
from abc import ABC

if typing.TYPE_CHECKING:
    import easyai.simulation.particles


class Message(ABC):
    """Used for type hinting. No methods at this time but may be expanded"""

    ...


@dataclass
class SimulationStep(Message):
    """Indicates the the simulation is stepping"""

    time: float


@dataclass
class SimulationState(Message):
    """Contains all the objects current in the simulation, as well as time"""

    time: float
    objects: Dict[str, simulation.particles.Particle] = field(
        default_factory=dict
    )


@dataclass
class ParticleAdded(Message):
    """Contains a particle added to the scenario"""

    time: float
    particle: simulation.particles.Particle


@dataclass
class TaskComplete(Message):
    """Indicates a particle that has had a task completed"""

    time: float
    particle: simulation.particles.Particle


@dataclass
class Collision(Message):
    """Indicates two particles that have collided
    by entering both or the other particle's zone"""

    time: float
    particles: List[simulation.particles.Particle] = field(
        default_factory=list
    )
