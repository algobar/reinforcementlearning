"""Module to define functions that can process state"""
from collections import deque
from easyai.environments.types import StateInfo

def initialize_state_queue(size: int) -> deque:
    """Returns a deque to track previous states"""
    return deque(maxlen=size)

def add_state(stored_state: deque, state: StateInfo) -> StateInfo:
    """Updates the deque with the new state"""
    stored_state.append(state)
    return state

def process_bearing_between(state: StateInfo, source: str, target: str) -> StateInfo:
    ...

def process_distance_between(state: StateInfo, source: str, target: str) -> StateInfo:
    ...