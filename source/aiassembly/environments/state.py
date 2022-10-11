"""
Module to define how state is represented

"""

from typing import List
import numpy

from collections import deque
from aiassembly.environments.types import StoredStateInfo, State


def initialize_state_history(size: int) -> StoredStateInfo:
    """Returns a deque to track previous states"""
    return deque(maxlen=size)


def add_state(state: State, stored_state: deque) -> StoredStateInfo:
    """Updates the deque with the new state"""
    stored_state.append(state)
    return stored_state

def get_state_history(stored_state: deque) -> List[State]:
    """Return the state history as an object"""
    return list(stored_state)