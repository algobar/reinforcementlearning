from functools import partial
from typing import Callable
import numpy
from easyai.library.vectors import magnitude, straight_line_path_2d
from easyai.simulation.particles import Particle


def go_to_point_2d(
    particle: Particle,
    destination: numpy.array,
    speed: float,
    timestep: float,
    threshold: float = 0.1,
    **kwargs
) -> bool:

    if magnitude(particle.position - destination) < threshold:
        return True

    particle.position = straight_line_path_2d(
        particle.position, destination, speed, timestep
    )
    return False


def remain_in_location_seconds(
    end_time: float, current_time: float, **kwargs
) -> bool:

    if current_time > end_time:
        return True

    return False


def create_behavior(behavior: Callable, **kwargs):

    return partial(behavior, **kwargs)
