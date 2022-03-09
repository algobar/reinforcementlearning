import numpy
from simulation.calculations import magnitude, straight_line_path_2d
from simulation.particles import Particle


def go_to_point_2d(
    particle: Particle,
    destination: numpy.array,
    speed: float,
    timestep: float,
    threshold: float = 0.1,
) -> bool:

    while magnitude(particle.position - destination) > threshold:
        particle.position = straight_line_path_2d(
            particle.position, destination, speed, timestep
        )

        yield False

    yield True


def remain_in_location_seconds(
    duration_seconds: float, timestep: float
) -> bool:

    time_waited: float = 0

    while time_waited < duration_seconds:

        time_waited += timestep

        yield False

    yield True
