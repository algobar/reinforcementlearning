from functools import partial
from typing import Dict, List, Optional
from easyai.environments.registry import register_function
import numpy
from easyai.particle_simulator.types import SIMULATION_NAME, Simulator
from easyai.environments.types import (
    EntityType,
    Parameter,
    Position,
    ResetSimulation,
    Task,
)
from easyai.particle_simulator.types import Particle, GetResetFunction

# particle specific functions --------------------------------


def create_position(x: float, y: float, z: float) -> Position:
    """Creates a new position given the initial coordinates"""
    return numpy.array([x, y, z], dtype=numpy.float32)


def create_particle(
    name: str,
    type: EntityType,
    radius: float,
    position: Optional[numpy.ndarray] = None,
    speed: Optional[float] = None,
) -> Particle:

    if position is None:

        position = create_position(0, 0, 0)

    speed = 0 if speed is None else speed

    return Particle(name, position, type, speed, radius)


def get_particle(simulator: Simulator, name: str) -> Particle:
    """Returns the particle of the given name"""

    return simulator.objects[name]


# simulator specific instructions ----------------------------------


def remove_particle(simulator: Simulator, name: str) -> Simulator:
    """Removes the provided name from the sim"""
    simulator.objects.pop(name)


def add_particle(simulator: Simulator, particle: Particle) -> Simulator:
    """Adds a particle to the simulation.

    :param particle: particle object
    :type particle: Particle
    :raises KeyError: If name exists
    """

    if particle.name in simulator.objects:
        raise KeyError(f"{particle.name} already exists!")

    simulator.objects[particle.name] = particle


def remove_all_particles(simulator: Simulator) -> Simulator:
    """
    Remove all the objects from the sim,
    """
    simulator.objects.clear()


def remove_all_scripts(simulator: Simulator) -> Simulator:

    simulator.scripts.clear()


def remove_all(simulator: Simulator) -> Simulator:

    remove_all_particles(simulator)
    remove_all_scripts(simulator)


def register_script(simulator: Simulator, script) -> Simulator:
    """Adds a callable script to the simulation
    to call and maintain

    :param script: [description]
    :type script: Script
    """
    simulator.scripts.append(script)


def update(simulator: Simulator, tasks: List[Task]) -> Simulator:
    """Updates the objects given the delta timestep in seconds"""

    for obj in simulator.objects.values():

        obj.update(timestep=simulator.step_size)

    for script in simulator.scripts:
        script.update(timestep=simulator.step_size, simulator=simulator)

    simulator.time += simulator.step_size


# reset specific instructions ------------------------


def parse_new_particle(data: Dict[str, float]) -> Particle:
    """Creates a new particle from the given data dict"""
    name = data["name"]
    ent_type = data["type"]
    position = data["x"], data["y"], data["z"]
    speed = data["speed"]
    radius = data["radius"]

    pos_arr = create_position(*position)
    new_particle = create_particle(name, ent_type, radius, pos_arr, speed)

    return new_particle


def reset_new_particle(simulator: Simulator, data: Dict[str, float]):
    """Implements a reset function to add a new particle"""
    particle = parse_new_particle(data)
    add_particle(simulator, particle)


def reset_simulation(
    parameters: List[Parameter], simulator: Simulator, get_param_func: GetResetFunction
) -> Simulator:
    """Resets the simulation with the provided parameters"""
    for param in parameters:
        param_type = param["parameter_type"]
        param_func = get_param_func(param_type)
        param_func_partial = partial(param_func, simulator)
        param_func_partial(param)


def create_reset_function(
    simulator: Simulator, get_param_func: GetResetFunction
) -> ResetSimulation:

    return partial(reset_simulation, simulator=simulator, get_param_func=get_param_func)


register_function("particle", SIMULATION_NAME, reset_new_particle)
