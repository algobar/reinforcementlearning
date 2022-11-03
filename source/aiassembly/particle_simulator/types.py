from dataclasses import dataclass, field
from typing import Any, Callable, Dict
from aiassembly.environments.types import (
    EntityType,
    EntityName,
    ImplementParameter,
    Parameter,
    Seconds,
)

import numpy

SIMULATION_NAME = "particle_simulation_environment"

SimulationTime = Seconds
Agent = EntityType("agent")
Base = EntityType("base")
Enemy = EntityType("enemy")


@dataclass
class Particle:
    """Represents a simple object with position and speed"""

    name: EntityName
    type: EntityType
    radius: float
    position: numpy.ndarray = field(
        default_factory=lambda: numpy.array([0, 0, 0])
    )
    speed: numpy.ndarray = field(
        default_factory=lambda: numpy.array([0, 0, 0])
    )


@dataclass
class Simulator:
    """
    Simulation that maintains and updates
    particles on a fixed timestep.

    Supports ability to add fixed scripts
    to manipulate objects
    """

    step_size: float
    time: float = 0
    objects: Dict[str, Particle] = field(default_factory=dict)


# Produces a matching function to implement the given parameter
ImplementParticleSimParameter = Callable[
    [Parameter, Simulator], ImplementParameter
]
GetResetFunction = Callable[[str], ImplementParticleSimParameter]
