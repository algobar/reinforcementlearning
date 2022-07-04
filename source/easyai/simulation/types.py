from enum import Enum, auto


class ParticleTypes(Enum):
    """Represents types of objects in the simulation"""

    AGENT = auto()
    BASE = auto()
    ENEMY = auto()
