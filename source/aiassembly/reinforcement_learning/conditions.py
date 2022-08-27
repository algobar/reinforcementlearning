""" 
Module to define conditions:

Conditions are used in RL to indicate when an agent is 'done' with the episode.

However, these conditions can also be used in other places, such as deciding
when the agent should take an action. Therefore, this is defined as a more
generic type to be used throughout the RL process.
"""
from typing import Callable, List, Dict

Condition = bool
ConditionFuncType = Callable[[object], Condition]

