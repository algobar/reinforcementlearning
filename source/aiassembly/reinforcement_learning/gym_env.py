"""
The following is an attempt to bring a more functional style approach to
the standard gym environment. Since the general policy interface includes
the step and reset function (along with space definitions), the intent
here is to provide as much external interface that allows the user
to set the callable functions, instead of having to override
the base multi agent environment.
"""
import dataclasses
from typing import Any, Callable, Dict, List, Protocol, Tuple

import gym
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, MultiEnvDict
from aiassembly.environments.types import (
    FixedAdvanceSimulation,
    ResetSimulation,
    ProcessStateFunc,
    State,
    StoredStateInfo,
    Task,
)
from aiassembly.reinforcement_learning.types import (
    QueryAgentFunc,
    FeatureProcessFunc,
    DoneProcessFunc,
    RewardProcessFunc,
)

class StateProtocol(Protocol):

    def add(self, state) -> None:
        ...

    def reset(self) -> None:
        ...

@dataclasses.dataclass(frozen=True)
class BaseMultiAgentEnv(MultiAgentEnv):

    observation_space: gym.Space
    action_space: gym.Space

    build_tasks: Callable[[StoredStateInfo, Dict[str, Any]], List[Task]]
    advance_simulation: FixedAdvanceSimulation
    reset_simulation: ResetSimulation
    query_agent_func: QueryAgentFunc
    process_state: ProcessStateFunc
    feature_processing: FeatureProcessFunc
    done_processing: DoneProcessFunc
    reward_processing: RewardProcessFunc

    # intended to be the only 'stateful'
    # portion of the class, besides the simulation
    state: StateProtocol

    def reset(self) -> MultiAgentDict:
        """Reset the simulation to its starting state"""
        self.state.reset()
        state = self.reset_simulation()
        state = self.process_state(state)
        self.state.add(state)
        _, query_status = self.query_agent_func(self.state)
        obs, obs_info = self.feature_processing(self.state, query_status)

        return obs

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """Advance the simulation with the given actions"""

        tasks = self.build_tasks(action_dict, self.stored_states)
        state = self.advance_simulation(tasks)
        self.state = self.process_state(state)
        should_query, query_status = self.query_agent_func(self.state)
        while not should_query:
            state = self.advance_simulation()
            self.state = self.process_state(state)
            should_query, query_status = self.query_agent_func(self.state)

        dones, _ = self.done_processing(self.state, query_status)
        rewards, _ = self.reward_processing(self.state, query_status)
        observations, _ = self.feature_processing(self.state, query_status)

        return observations, rewards, dones, {}
