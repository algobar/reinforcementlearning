"""Module for building generic functional patterns"""

import functools
from typing import Callable


def pass_along(intial_value, *chain_of_funcs):
    """This defines a generic pattern of taking a starting value, 
    passing it into another function, and continuing until the end
    
    such as....
    
    initial = 0
    
    pass_along(0, add_one, add_two, add_three)
    
    where the value at each operation becomes....
    
    1, 3, 6
    
    """
    
    def evaluate_function(func: Callable[[object], object], value: object):
        
        return func(value)
    
    result = functools.reduce(evaluate_function, chain_of_funcs, intial_value)
    
    return result