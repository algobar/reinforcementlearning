from typing import Any, Callable, Dict, List, NamedTuple, Set, NewType, Iterable, Tuple, TypeVar

import numpy

ENVIRONMENT_BUILDER = "__build_environment"

T = TypeVar("T")
# this is for data specific to state of the simulation
EntityType = NewType("EntityType", str)
EntityName = NewType("EntityName", str)
Position = numpy.ndarray
Speed = NewType("speed", float)

class Position(NamedTuple):
    x: float
    y: float
    z: float
class Entity(NamedTuple):
    name: str 
    type: str
    position: Position
    speed: float


# Types used to communicate back and forth with
# simulation
Seconds = NewType("seconds", float)
Parameter = NewType("Parameter", Dict[str, Any])
Task = List[Parameter]

# function that takes a parameter and implements it
# in the given simulation
ImplementParameter = Callable[[Parameter], None]
ImplementTask = Callable[[Task], None]

# these define expectations for accepting StateInfo,
# transforming, combining, etc., and returning a new one
StateInfo = NewType("StateInfo", Any)
StoredStateInfo = Iterable[StateInfo]
# generic function that takes in state data and
# modifies/transforms/etc
ProcessStateFunc = Callable[[StateInfo], StateInfo]

# this defines our interaction with the simulation
# these would typically be methods within an interface.
SimulationConfig = Dict[str, Any]
ResetSimulation = Callable[[List[Parameter]], StateInfo]
AdvanceSimulation = Callable[[List[Task], Seconds], StateInfo]

class SimulationInterface(NamedTuple):
    """General Interface for communicating with simulations"""
    reset_simulation: ResetSimulation
    advance_simulation: AdvanceSimulation

# defines building a simulation via a configuration
BuildSimulation = Callable[[SimulationConfig], SimulationInterface]


FunctionRegistry = Dict[str, Callable[[Any], Any]]
