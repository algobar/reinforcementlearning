from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
from aiassembly.environments.interface import register_environment_interface
from aiassembly.library.registry import (
    get_registered_function,
    register_function,
)
import numpy
from aiassembly.particle_simulator.types import (
    SIMULATION_NAME,
    ImplementParticleSimParameter,
    Simulator,
)
from aiassembly.environments.types import (
    EntityType,
    Parameter,
    ResetSimulation,
    Seconds,
    SimulationInterface,
    Task,
)
from aiassembly.particle_simulator.types import Particle, GetResetFunction

# particle specific functions --------------------------------


def create_position(x: float, y: float, z: float) -> numpy.ndarray:
    """Creates a new position given the initial coordinates"""
    return numpy.array([x, y, z], dtype=numpy.float32)


# simulator specific instructions ----------------------------------


def remove_particle(simulator: Simulator, name: str) -> Simulator:
    """Removes the provided name from the sim"""
    simulator.objects.pop(name)
    return Simulator


def add_particle(simulator: Simulator, particle: Particle) -> Simulator:
    """Adds a particle to the simulation.

    :param particle: particle object
    :type particle: Particle
    :raises KeyError: If name exists
    """

    if particle.name in simulator.objects:
        raise KeyError(f"{particle.name} already exists!")

    simulator.objects[particle.name] = particle

    return simulator


def set_speed_vector(
    simulator: Simulator, name: str, x: float, y: float, z: float
) -> None:
    """Set a speed vector for the object"""
    part = simulator.objects.get(name)
    speed_vector = create_position(x, y, z)
    part.speed = speed_vector


def remove_all(simulator: Simulator) -> Simulator:
    """Remove all object from sim"""
    simulator.objects.clear()

    return simulator


def step_simulator(simulator: Simulator) -> Simulator:
    """Updates the objects given the delta timestep in seconds"""

    for part in simulator.objects.values():
        part.position = part.position + part.speed * simulator.step_size

    simulator.time += simulator.step_size

    return simulator


def apply_tasks(simulator: Simulator, tasks: List[Parameter]) -> Simulator:
    """Update the objects, new scripts, etc. with new tasks"""

    for task in tasks:
        task_type = task.pop("type")
        task_func = get_registered_function(task_type, SIMULATION_NAME)
        task_func(simulator, **task)

    return simulator


def run_simulation(
    tasks: List[Task], advance: Seconds, get_state_func, simulator: Simulator
) -> Dict[str, Any]:
    """Implements any actions and steps the simulator at the given rate"""
    start_time = simulator.time
    apply_tasks(simulator, tasks)
    while simulator.time < start_time + advance:
        step_simulator(simulator)

    return get_state_func(simulator)


# reset & setup specific instructions ------------------------


def parse_new_particle(data: Dict[str, float]) -> Particle:
    """Creates a new particle from the given data dict"""
    name = data["name"]
    ent_type = data["type"]
    pos_data = data["position"]
    vel_data = data["velocity"]
    position = pos_data["x"], pos_data["y"], pos_data["z"]
    velocity = vel_data["x"], vel_data["y"], vel_data["z"]
    radius = data["radius"]

    pos_arr = create_position(*position)
    vel_arr = create_position(*velocity)
    new_particle = Particle(
        name=name,
        type=ent_type,
        radius=radius,
        position=pos_arr,
        speed=vel_arr,
    )

    return new_particle


def reset_new_particle(data: Dict[str, float], simulator: Simulator):
    """Parses the particle data and adds to the simulator"""
    particle = parse_new_particle(data)
    add_particle(simulator, particle)


def get_reset_function(name: str) -> ImplementParticleSimParameter:
    """Get the provided reset function"""
    return get_registered_function(name, SIMULATION_NAME)


def reset_simulation(
    parameters: List[Parameter],
    simulator: Simulator,
    get_param_func: GetResetFunction,
    get_state_func: Callable[[Simulator], Dict[str, Any]],
) -> Dict[str, Any]:

    remove_all(simulator)
    """Resets the simulation with the provided parameters"""
    for param in parameters:
        param_type = param["parameter_type"]
        param_func = get_param_func(param_type)
        param_func_partial = partial(param_func, simulator=simulator)
        param_func_partial(param)

    return get_state_func(simulator)


def create_reset_function(
    simulator: Simulator,
    get_param_func: GetResetFunction,
    get_state_func: Callable[[Simulator], Dict[str, Any]],
) -> ResetSimulation:
    """Creates the reset simulation function that matches the
    common environment interface
    """
    return partial(
        reset_simulation,
        simulator=simulator,
        get_param_func=get_param_func,
        get_state_func=get_state_func,
    )


def get_all_entities(simulator: Simulator) -> List[str]:
    """Get all entities in simulation"""
    return list(simulator.objects.keys())


def get_named_entity_position(
    name: str, simulator: Simulator
) -> Tuple[float, float, float]:

    pos = simulator.objects.get(name).position

    return {"x": pos[0], "y": pos[1], "z": pos[2]}


def get_simulation_state(
    simulator: Simulator,
    entity_func: Callable[[Simulator], List[str]],
    data_funcs: List[Callable[[str, Simulator], Dict[str, Any]]],
):
    """Get the simulation state for the entities returned by the func, with the data obtained by the callable"""
    entities = entity_func(simulator)
    output = {}

    for entity in entities:
        output[entity] = {}
        for each_data_func in data_funcs:
            output[entity].update(each_data_func(entity, simulator))

    return output


def get_requested_state(
    simulator: Simulator,
    state_funcs: List[Callable[[Simulator], Dict[str, Any]]],
) -> Dict[str, Any]:
    """Gets the requested state for each unique state gettr func"""
    output = {}

    for get_state in state_funcs:
        output.update(get_state(simulator))

    return output


def create_state_function(
    state_request: List[Dict[str, str]]
) -> Callable[[None], List[Dict[str, Any]]]:
    """Function that parses the sim based on given request and returns data"""

    state_funcs = []

    for each_request in state_request:
        entity_func_name = each_request["select"]
        entity_func = get_registered_function(
            f"select_func_{entity_func_name}", SIMULATION_NAME
        )

        data_funcs = []
        for request in each_request["properties"]:
            data_funcs.append(
                get_registered_function(request["name"], SIMULATION_NAME)
            )

        sim_state_func = partial(
            get_simulation_state,
            entity_func=entity_func,
            data_funcs=data_funcs,
        )

        state_funcs.append(sim_state_func)

    return partial(get_requested_state, state_funcs=state_funcs)


def build_environment(
    step_size: float, state_request: List[Dict[str, str]]
) -> SimulationInterface:
    """Initializes the environment and its interfaces"""
    sim = Simulator(step_size)
    state_func = create_state_function(state_request)
    reset_func = create_reset_function(sim, get_reset_function, state_func)
    advance_func = partial(
        run_simulation, simulator=sim, get_state_func=state_func
    )

    return SimulationInterface(reset_func, advance_func)

print(f"****loading {SIMULATION_NAME}****")
register_environment_interface(SIMULATION_NAME, build_environment)
register_function("particle", SIMULATION_NAME, reset_new_particle)
register_function("select_func_all", SIMULATION_NAME, get_all_entities)
register_function("position", SIMULATION_NAME, get_named_entity_position)
register_function("set_speed_vector", SIMULATION_NAME, set_speed_vector)
