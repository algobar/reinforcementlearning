"""
Module to define how state is represented

"""

from dataclasses import dataclass, field
from typing import List
import numpy

from collections import deque
from aiassembly.environments.types import StoredStateInfo, State

@dataclass
class StateData:

    size: int
    states: deque = None

    def __post_init__(self):
        """Initialize the deque once size is given"""
        self.states = deque(maxlen=self.size)

    def append(self, state):
        """Add new state"""
        self.states.append(state)
    
    def reset(self):
        """Erase all existing states"""
        self.states.clear()