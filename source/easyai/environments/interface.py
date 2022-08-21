from easyai.environments.types import (
    ENVIRONMENT_BUILDER,
    BuildSimulation,
    SimulationConfig,
    SimulationInterface
)

from easyai.environments.registry import get_registered_function, register_function


def register_environment_interface(owner: str, build_sim_func: BuildSimulation) -> None:
    """This function is used for simulators and other data
    generation means to provide the necessary components to
    register the functions needed to generate their environment,
    as well as provide the step and reset functions needed at
    runtime to pass data back and forth
    """

    register_function(ENVIRONMENT_BUILDER, owner, build_sim_func)

def load_environment_interface(sim_config: SimulationConfig, owner: str) -> SimulationInterface:
    """Loads the simulation builder and initializes it with the given config"""
    builder = get_registered_function(ENVIRONMENT_BUILDER, owner)
    return builder(**sim_config)