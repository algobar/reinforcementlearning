from dataclasses import dataclass, field
from typing import Callable, Dict, List, NamedTuple
from aiassembly.environments.types import (
    EntityType,
    EntityName,
    ImplementParameter,
    Parameter,
    Position,
    Seconds,
    Speed,
)

SIMULATION_NAME = "particle_simulation_environment"

SimulationTime = Seconds
Agent = EntityType("agent")
Base = EntityType("base")
Enemy = EntityType("enemy")
# Functions that implement sim specific behavior
Behavior = Callable[[SimulationTime], None]
Script = Callable[[SimulationTime], None]


class Particle(NamedTuple):
    """Represents a simple object with position and speed"""

    name: EntityName
    position: Position
    type: EntityType
    speed: Speed
    radius: float


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
    scripts: List[Callable] = field(default_factory=list)

# Produces a matching function to implement the given parameter
ImplementParticleSimParameter = Callable[
    [Parameter, Simulator], ImplementParameter
]
GetResetFunction = Callable[[str], ImplementParticleSimParameter]
