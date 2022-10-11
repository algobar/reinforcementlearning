"""
The following is an attempt to bring a more functional style approach to
the standard gym environment. Since the general policy interface includes
the step and reset function (along with space definitions), the intent
here is to provide as much external interface that allows the user
to set the callable functions, instead of having to override
the base multi agent environment.
"""
import dataclasses
from typing import Tuple

import gym
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, MultiEnvDict
from aiassembly.environments.types import (
    FixedAdvanceSimulation,
    ResetSimulation,
    ProcessStateFunc,
    StoredStateInfo,
)
from aiassembly.reinforcement_learning.types import (
    QueryAgentFunc,
    FeatureProcessFunc,
    DoneProcessFunc,
    RewardProcessFunc,
)


@dataclasses.dataclass
class BaseMultiAgentEnv(MultiAgentEnv):

    observation_space: gym.Space
    action_space: gym.Space

    build_tasks: ...
    advance_simulation: FixedAdvanceSimulation
    reset_simulation: ResetSimulation
    query_agent_func: QueryAgentFunc
    process_state: ProcessStateFunc
    feature_processing: FeatureProcessFunc
    done_processing: DoneProcessFunc
    reward_processing: RewardProcessFunc

    stored_states: StoredStateInfo

    def reset(self) -> MultiAgentDict:
        """Reset the simulation to its starting state"""
        state = self.reset_simulation()
        self.stored_states = self.process_state(state)
        _, query_status = self.query_agent_func(self.stored_states)
        obs, obs_info = self.feature_processing(self.stored_states, query_status)

        return obs

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """Advance the simulation with the given actions"""

        tasks = self.build_tasks(action_dict, self.stored_states)
        state = self.advance_simulation(tasks)
        self.stored_states = self.process_state(state)
        should_query, query_status = self.query_agent_func(self.stored_states)
        while not should_query:
            state = self.advance_simulation()
            self.stored_states = self.process_state(state)
            should_query, query_status = self.query_agent_func(self.stored_states)

        dones, _ = self.done_processing(self.stored_states, query_status)
        rewards, _ = self.reward_processing(self.stored_states, query_status)
        observations, _ = self.feature_processing(self.stored_states, query_status)

        return observations, rewards, dones, {}
