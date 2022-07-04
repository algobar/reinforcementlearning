from typing import Any, Callable, List, OrderedDict, Union
from gym import spaces

import numpy

from . import calculations


BOX_LOW: float = -5
BOX_HIGH: float = 5
DEFAULT_INVALID: float = -2
DEFAULT_DTYPE = numpy.float32

'''
How to define an observation? 
1. Defines the space for the value
2. Defines how the value is extracted and fit into space definition
3. Defines how the result is then modified for network

'''

# Base functions

ExtractorFunc = Callable[[object], numpy.ndarray]
ModifierFunc = Callable[[numpy.ndarray], numpy.ndarray]

def create_array(data: List[Union[float, int, numpy.array]], dtype=DEFAULT_DTYPE) -> numpy.ndarray:
    """Create a numpy array from the given list of data. Provides default data type"""
    return numpy.array(data, dtype=dtype)

def define_box_space(low_values: List[float], high_values: List[float], dtype=DEFAULT_DTYPE) -> spaces.Discrete:
    """Create a box space with the given low and high values for each index"""
    return spaces.Box(low_values, high_values, dtype=dtype)

def define_discrete_space(count: int) -> spaces.Discrete:
    """Create a generic discrete space"""
    return spaces.Discrete(count)

def define_dict_space(space_dict: OrderedDict[str, spaces.Space]) -> spaces.Dict:
    """Create a generic dict type space"""
    return spaces.Dict(space_dict)

def add_space_to_dict(name: str, spaces: spaces.Space, space_dict: OrderedDict[str, spaces.Space]) -> OrderedDict[str, spaces.Space]:
    """Add the space with the given name to the collection"""
    space_dict[name] = spaces
    
    return space_dict

def create_observation(state, extractor_func: ExtractorFunc, modifier_func: ModifierFunc) -> numpy.ndarray:
    """Create an observation by grabbing it from the state and manipulation it accordingly"""
    observation = extractor_func
    modified = modifier_func(observation)
    
    return modified

# Generic space definitions
def define_sin_cos_space() -> spaces.Box:
    """Define a 2D space representing an angle broken into its sin/cos components"""
    
    dimensions = 2
    sin_cos_min = -1
    sin_cos_max = 1
    
    return define_box_space([sin_cos_min]*dimensions, [sin_cos_max]*2)

def define_action_mask_space(total_actions: int) -> spaces.Box:
    """Define a space representing an action mask used to force agent to pick only certain ones"""
    return define_box_space([0]*total_actions, [1]*total_actions)

# Modifier Functions

def convert_angle_to_sin_cos(angle_radians: numpy.ndarray) -> numpy.ndarray:
    """Convert the angle to a 2D sin/cos representation"""
    return create_array([numpy.cos(angle_radians), numpy.sin(angle_radians)])