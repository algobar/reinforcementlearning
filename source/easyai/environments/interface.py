from typing import List
from easyai.environments.types import (
    AdvanceSimulation,
    Parameter,
    ResetSimulation,
    Seconds,
    StateInfo,
    StoredStateInfo,
    Task,
    TransformState,
    UpdateStateInfo,
)


def reset_environment(
    parameters: List[Parameter],
    stored_states: StoredStateInfo,
    reset_sim: ResetSimulation,
    update_state: UpdateStateInfo,
    transform_state: TransformState,
) -> StateInfo:
    """Resets the environment by passing the new parameters
    and obtaining the new state"""

    new_state = reset_sim(parameters)
    new_stored_states = update_state(new_state, stored_states)
    formed_state = transform_state(new_stored_states)

    return formed_state


def step_environment(
    tasks: List[Task],
    update_time: Seconds,
    stored_states: StoredStateInfo,
    advance_sim: AdvanceSimulation,
    update_state: UpdateStateInfo,
    transform_state: TransformState,
) -> StateInfo:
    """Advances the environment by the given time"""

    new_state = advance_sim(tasks, update_time)
    new_stored_states = update_state(new_state, stored_states)
    formed_state = transform_state(new_stored_states)

    return formed_state
