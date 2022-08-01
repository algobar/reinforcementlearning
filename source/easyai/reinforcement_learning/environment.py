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


def set_data_interface(
    environment: BaseMultiAgentEnv, data_interface: Callable
) -> BaseMultiAgentEnv:

    environment.data_interface
    return environment


def set_observation_space(
    environment: BaseMultiAgentEnv, obs_space: gym.Space
) -> BaseMultiAgentEnv:

    environment.observation_space = obs_space
    return environment


def set_action_space(
    environment: BaseMultiAgentEnv, action_space: gym.Space
) -> BaseMultiAgentEnv:

    environment.action_space = action_space
    return environment


def set_observation_func(
    environment: BaseMultiAgentEnv, observation_func: Callable[[object], Dict[str, Any]]
) -> BaseMultiAgentEnv:

    environment.observations = observation_func
    return environment


def set_action_func(
    environment: BaseMultiAgentEnv, action_func: Callable[[MultiAgentDict], None]
) -> BaseMultiAgentEnv:

    environment.actions = action_func
    return environment


def set_done_func(
    environment: BaseMultiAgentEnv,
    done_func: Callable[[MultiAgentDict], MultiAgentDict],
) -> BaseMultiAgentEnv:

    environment.dones = done_func
    return environment


def set_reward_func(
    environment: BaseMultiAgentEnv,
    reward_func: Callable[[MultiAgentDict], MultiAgentDict],
) -> BaseMultiAgentEnv:

    environment.dones = reward_func
    return environment


def set_step_func(environment: BaseMultiAgentEnv, step_func: StepFunction):
    environment.step_func = step_func
    return environment


def set_reset_func(environment: BaseMultiAgentEnv, reset_func: ResetFunction):
    environment.reset_func = reset_func
    return environment


def default_step_function(
    action_dict: MultiAgentDict,
    observation_func: Callable,
    action_func: Callable,
    rewards_func: Callable,
    done_func: Callable,
    update_environment_func: Callable[[None], object],
    get_ready_agents_func: Callable[[None], List[str]],
) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:

    action_func(action_dict)
    state = update_environment_func()
    ready_agents = get_ready_agents_func()

    while len(ready_agents) == 0:
        state = update_environment_func()
        ready_agents = get_ready_agents_func()

    dones = done_func(state)
    assert "__all__" in dones

    rewards = rewards_func(ready_agents)
    observations = observation_func(ready_agents)
    info = {}

    return observations, rewards, dones, info


def build_gym_interface(
    observation_space,
    action_space,
    observation_func,
    action_func,
    reward_func,
    done_func,
    step_func,
    reset_func,
) -> BaseMultiAgentEnv:

    env = BaseMultiAgentEnv()

    env = set_observation_space(env, observation_space)
    env = set_action_space(env, action_space)
    env = set_observation_func(env, observation_func)
    env = set_action_func(env, action_func)
    env = set_reward_func(env, reward_func)
    env = set_done_func(env, done_func)
    env = set_step_func(env, step_func)
    env = set_reset_func(env, reset_func)

    return env
