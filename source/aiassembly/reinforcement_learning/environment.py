"""
The following is an attempt to bring a more functional style approach to
the standard gym environment. Since the general policy interface includes
the step and reset function (along with space definitions), the intent
here is to provide as much external interface that allows the user
to set the callable functions, instead of having to override
the base multi agent environment.
"""
import dataclasses
from typing import Any, Callable, Dict, List, Tuple

import gym
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, MultiEnvDict

ResetFunction = Callable[[None], MultiEnvDict]
StepFunction = Callable[
    [MultiAgentDict],
    Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict],
]


@dataclasses.dataclass
class BaseMultiAgentEnv(MultiAgentEnv):

    data_interface: object = None
    observation_space: gym.Space = None
    action_space: gym.Space = None

    observations: Callable = None
    actions: Callable = None
    dones: Callable = None
    rewards: Callable = None
    reset_func: ResetFunction = None
    step_func: StepFunction = None

    def reset(self) -> MultiAgentDict:

        return self.reset_func(observation_func=self.observations)

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:

        return self.step_func(
            action_dict=action_dict,
            observation_func=self.observations,
            action_func=self.actions,
            reward_func=self.rewards,
            done_func=self.dones,
        )