import numpy
from easyai.library.vectors import magnitude, straight_line_path_2d
from easyai.particle_simulator.types import Particle


def go_to_point_2d(
    particle: Particle,
    destination: numpy.array,
    speed: float,
    timestep: float,
    threshold: float = 0.1,
) -> None:

    if magnitude(particle.position - destination) < threshold:
        return

    particle.position = straight_line_path_2d(
        particle.position, destination, speed, timestep
    )


def remain_in_location_seconds(end_time: float, current_time: float) -> bool:

    if current_time > end_time:
        return

    return False
