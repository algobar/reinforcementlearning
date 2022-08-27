"""Module to define how state is represented

End goal: State can be represented in a type
enforced manner. We can track the features
in a tree represented manner if there 
exists a many to one feature set.

State can be squashed into a flat array
where it can easily be passed to different
functions to perform matrix based operations
where applicable. 

Focus on pulling only raw data...
Remove calculations that can be done at 
the extraction level when it comes
to relationship specific data
or data over time (unless specified)

- entity_1 (feature collection)
    - self_state (feature branch)
        - type: index representing type (leaf)
        - pos_x: float (leaf)
        - pos_y: float (leaf)
        - pos_z: float (leaf)
    - sub_component_x (feature branch)
        - component_type: index (leaf)
        - f1: float (leaf)
        - f2: float (leaf)
    - sub_component_y (feature branch)
        - ... (leafs)
    - sensed_entity_x (feature branch)
        - type: int (feature)
        - source: int representing sub-component (feature)
        - pos_x: float (leaf)
        - pos_y: float (leaf)
- entity_2 (feature collection)
    - type: index representing type
    - pos_x: float
    - pos_y: float
    - pos_z: float

if we look at a tree structure,
we can organize things into (3) categories
- feature collection - represents one or more feature branch
- feature branch - represents one or more features
- features - float/int representing data

The important part is that we understand that 
we don't leave dangling features outside of a branch.

If we call 'flatten' on the entity_1, we should see the features reduced 
to a matrix:
                | type | pos_x | pos_y | pos_z | 
entity_1        |  0   |   0   |   1   |   2   |
sensed_entity_1 |  1   |   3   |   4   |   5   |

To define this groups, the gym environment
needs to list out the properties to measure.

{
    "team": [
        "state": {
            "pos_x",
            "pos_y",
            "pos_z",
        }
        "subcomponent_type": {
            "feat1",
            "feat2",
            ...
        }
        "sensed_entities": {
            "pos_x",
            "pos_y",
            "pox_z"
        }
    ]

}

The root level corresponds to an entity
type. This type definition is loose at 
the moment. But can separate data based
on different object types: (i.e car vs
airplane).

Based on the grouping above, for each
type where there exists a repeated 
relationship, would drive multiple
resulting feature branches



"""

import numpy

from collections import deque
from aiassembly.environments.types import Feature, FeatureBranch, State


def initialize_state_queue(size: int) -> deque:
    """Returns a deque to track previous states"""
    return deque(maxlen=size)


def add_state(stored_state: deque, state: State) -> State:
    """Updates the deque with the new state"""
    stored_state.append(state)
    return state



