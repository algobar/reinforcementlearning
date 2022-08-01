from typing import Any, Callable, Dict, List, NamedTuple, Set, NewType, Iterable

import numpy

# this is for data specific to state of the simulation
EntityType = NewType("EntityType", str)
EntityName = NewType("EntityName", str)
Position = numpy.ndarray
Speed = NewType("speed", float)

# Types used to communicate back and forth with
# simulation
Seconds = NewType("seconds", float)
Parameter = NewType("Parameter", Dict[str, Any])
Task = List[Parameter]

# function that takes a parameter and implements it
# in the given simulation
ImplementParameter = Callable[[Parameter], None]
ImplementTask = Callable[[Task], None]

StateInfo = NewType("StateInfo", Dict[str, Any])
StoredStateInfo = Iterable[StateInfo]
UpdateStateInfo = Callable[[StateInfo, StoredStateInfo], StoredStateInfo]

# this defines our interaction with the simulation
# these would typically be methods within an interface.
ResetSimulation = Callable[[List[Parameter]], StateInfo]
AdvanceSimulation = Callable[[List[Task], Seconds], StateInfo]

# these define expectations for accepting StateInfo,
# transforming, combining, etc., and returning a new one
TransformState = Callable[[StoredStateInfo], StateInfo]


FunctionRegistry = Dict[str, Callable[[Any], Any]]
