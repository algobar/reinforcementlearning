"""Module to define functions that extract data out of objects to be used elsewhere."""

# extracting a single value
from functools import partial
from typing import Any, Callable, Dict, List
from aiassembly.environments.types import State

def get_distance_between_points(state: State, originator: str, target: str) -> float:
    """Return the distance between the two entities"""
    return 0.0


def get_relative_bearing_between(state: State, originator: str, target: str) -> float:
    """Return the heading of the"""
    return 0.0


def get_speed(state: State, originator: str) -> float:
    """Get the speed of the object"""
    return 0.0


#
# Functions that extract from a series of values
#


def select_entity_at_index(entities: List[str], index: int) -> str:
    """Get the entity at the index in the given list"""
    return entities[index]


def get_entities_of_type(state: Simulator, ptype: ParticleTypes) -> List[str]:
    """Get the entities matching the provided type"""

    return [part for part in state.objects if state.get(part).type == ptype]


def extract_relative_data(
    state,
    originator: str,
    relative_func: Callable[[Simulator], str],
    func_to_run: Callable[[Simulator, str, str], float],
) -> float:
    """Uses given func to extract entity and runs relative function"""
    relative_ent = relative_func(state)
    result = func_to_run(state, originator, relative_ent)

    return result


def select_list(
    state: Simulator,
    selection_func: Callable[[Simulator], List[str]],
    ordering_func: Callable[[Simulator, List[str]], List[str]],
) -> List[str]:
    """Selects a list by the the given func, then orders them, returning the final list"""
    names = selection_func(state)
    result = ordering_func(state, names)

    return result


def order_by_func(
    state: Simulator,
    entities: List[str],
    extractor_func: Callable[[Simulator, str], float]
) -> List[str]:
    """Orders the objects by their retrieved distances"""

    data = [(i, extractor_func(state, i)) for i in entities]
    result = sorted(data, key=lambda x: x[1])
    return result
