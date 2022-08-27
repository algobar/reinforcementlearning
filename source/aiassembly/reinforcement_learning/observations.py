from dataclasses import dataclass
from typing import Callable, Iterable, List, NamedTuple, OrderedDict, Union
from gym import spaces

import numpy

BOX_LOW: float = -5
BOX_HIGH: float = 5
DEFAULT_INVALID: float = -2
DEFAULT_DTYPE = numpy.float32
SIN_COS_DIMENSIONS = 2
SIN_COS_MIN = -1
SIN_COS_MAX = 1

"""
How to define an observation?
1. Defines the space for the value
2. Defines how the value is extracted and fit into space definition

"""


ExtractorFunc = Callable[[object], numpy.ndarray]
ModifierFunc = Callable[[numpy.ndarray], numpy.ndarray]


@dataclass
class Measurement:
    """Measurement defines the space, as well as how to obtain the data"""

    name: str
    space: spaces.Space
    extractor: ExtractorFunc

    def extract(self, state, entity: str):
        ...


# Base functions
def create_array(
    data: List[Union[float, int, numpy.array]], dtype=DEFAULT_DTYPE
) -> numpy.ndarray:
    """Create a numpy array from the given list of data. Provides default data type"""
    return numpy.array(data, dtype=dtype)


def define_box_space(
    low_values: List[float], high_values: List[float], dtype=DEFAULT_DTYPE
) -> spaces.Discrete:
    """Create a box space with the given low and high values for each index"""
    low_values = numpy.array(low_values)
    high_values = numpy.array(high_values)
    return spaces.Box(low_values, high_values, dtype=dtype)


def define_discrete_space(count: int) -> spaces.Discrete:
    """Create a generic discrete space"""
    return spaces.Discrete(count)


def define_tuple_space(measurement: Iterable[Measurement]) -> spaces.Tuple:
    """Repeats the space definition for given count"""
    result = spaces.Tuple([meas.space for meas in measurement])
    return result


def define_dict_space(measurement: OrderedDict[str, Measurement]) -> spaces.Dict:
    """Create a generic dict type space"""
    return spaces.Dict({m.name: m.space for m in measurement})


def repeat_space(space: spaces.Space, count: int) -> spaces.Tuple:

    return [space for _ in range(count)]


def add_space_to_dict(
    name: str, spaces: spaces.Space, space_dict: OrderedDict[str, spaces.Space]
) -> OrderedDict[str, spaces.Space]:
    """Add the space with the given name to the collection"""
    space_dict[name] = spaces

    return space_dict


def create_observation(state, extractor_func: ExtractorFunc) -> numpy.ndarray:
    """Create an observation by grabbing it from the state and manipulation it accordingly"""
    observation = extractor_func(state)
    return observation


# Generic space definitions
def define_sin_cos_space() -> spaces.Box:
    """Define a 2D space representing an angle broken into its sin/cos components"""

    return define_box_space([SIN_COS_MIN] * SIN_COS_DIMENSIONS, [SIN_COS_MAX] * 2)


def define_action_mask_space(total_actions: int) -> spaces.Box:
    """Define a space representing an action mask used to force agent to pick only certain ones"""
    return define_box_space([0] * total_actions, [1] * total_actions)


# Modifier Functions


def convert_angle_to_sin_cos(angle_radians: numpy.ndarray) -> numpy.ndarray:
    """Convert the angle to a 2D sin/cos representation"""
    return create_array([numpy.cos(angle_radians), numpy.sin(angle_radians)])
