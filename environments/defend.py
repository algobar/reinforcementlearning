from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, OrderedDict, Tuple
from attr import has
from gym import spaces
from ray.rllib.env import MultiAgentEnv
import numpy
from simulation.simulator import SimpleWorld
from simulation.behaviors import GoToPoint2D, RemainInLocationSeconds
from simulation.particles import (
    Particle,
    Types,
)
from simulation.scripts import CreateEntityInterval, Script
from simulation import calculations


BOX_LOW: float = -5
BOX_HIGH: float = 5
DEFAULT_INVALID: float = -2


def float64_array(value):
    """Creates a float 32 array"""
    return numpy.array([value], dtype=numpy.float64)


def order_enemies_by_distance(simulator: SimpleWorld, poi: Particle) -> List:

    enemy_distance = [
        (
            enemy,
            calculations.distance_between(simulator.get(enemy), poi),
        )
        for enemy in simulator.get_all_of_type(Types.ENEMY)
    ]

    sorted_enemies_by_distance = sorted(enemy_distance, key=lambda x: x[1])

    return [enemy[0] for enemy in sorted_enemies_by_distance]


class BaseCallable(ABC):
    def __init__(self, **kwargs) -> None:
        self.__name__ = self.__class__.__name__

    def __call__(self, **kwargs) -> Any:
        ...


class Modifier(BaseCallable):
    """Modify observations"""

    ...


class NormalizeBox(Modifier):
    @staticmethod
    def normalize_box(
        value: float,
        low: float,
        high: float,
        norm_low: float = -1.0,
        norm_high: float = 1.0,
    ):
        """Bound the given value given a box between specified low and high"""
        return numpy.interp(
            value,
            [low, high],
            [norm_low, norm_high],
        )

    def __call__(self, value: float, low: float, high: float, **kwargs) -> Any:
        return self.normalize_box(value, low, high)


class Observation(BaseCallable):
    def __init__(
        self, low: float, high: float, modifier: Modifier = None, **kwargs
    ) -> None:

        self.low = low
        self.high = high
        self.modifier = modifier
        super().__init__(**kwargs)

    def modify(self, value):
        """Modify the value if the modifier exists"""
        if not self.modifier:
            return value

        return self.modifier(value, self.low, self.high)

    @abstractmethod
    def space(self) -> spaces.Space:
        ...


class DistanceToPoint(Observation):
    def __init__(
        self,
        low: float = 0,
        high: float = 30,
        modifier: Modifier = None,
        **kwargs,
    ) -> None:
        super().__init__(low, high, modifier, **kwargs)

    def space(self) -> spaces.Box:

        return spaces.Box(
            low=numpy.array([BOX_LOW]),
            high=numpy.array([BOX_HIGH]),
            dtype=numpy.float64,
        )

    def __call__(
        self, name: str, other: str, simulator: SimpleWorld, **kwargs
    ) -> numpy.array:

        if name is None:
            return float64_array(DEFAULT_INVALID)

        distance = calculations.distance_between(
            simulator.get(name), simulator.get(other)
        )

        return float64_array(self.modifier(distance, self.low, self.high))


class AbsoluteBearing(Observation):
    def __init__(
        self,
        low: float = -1,
        high: float = 1,
        modifier: Modifier = None,
        **kwargs,
    ) -> None:
        super().__init__(low, high, modifier, **kwargs)

    def space(self) -> spaces.Box:

        return spaces.Box(
            low=numpy.array([BOX_LOW] * 2),
            high=numpy.array([BOX_HIGH] * 2),
            dtype=numpy.float64,
        )

    def __call__(
        self, name: str, other: str, simulator: SimpleWorld, **kwargs
    ) -> Any:

        if name is None:
            return numpy.array([DEFAULT_INVALID] * 2, numpy.float64)

        abs_bearing: float = calculations.absolute_bearing_between(
            simulator.get(name), simulator.get(other)
        )

        sin = numpy.sin(abs_bearing)
        cos = numpy.cos(abs_bearing)

        return numpy.array([cos, sin], numpy.float64)


class Speed(Observation):
    def __init__(
        self,
        low: float = 0,
        high: float = 30,
        modifier: Modifier = None,
        **kwargs,
    ) -> None:
        super().__init__(low, high, modifier, **kwargs)

    def space(self) -> spaces.Box:

        return spaces.Box(
            low=numpy.array([BOX_LOW]),
            high=numpy.array([BOX_HIGH]),
            dtype=numpy.float64,
        )

    def __call__(self, name: str, simulator: SimpleWorld, **kwargs) -> Any:

        if name is None:
            return float64_array(DEFAULT_INVALID)

        speed: float = simulator.get(name).speed

        return float64_array(self.modify(speed))


class ValidActions(Observation):
    def __init__(self, fixed: int, variable: int, **kwargs) -> None:

        self.fixed_actions = fixed
        self.variable_actions = variable

    def space(self) -> spaces.Space:
        return spaces.Box(
            low=numpy.array(
                [0] * (self.fixed_actions + self.variable_actions)
            ),
            high=numpy.array(
                [1] * (self.fixed_actions + self.variable_actions)
            ),
            dtype=numpy.float64,
        )

    def __call__(self, variable_list: List, **kwargs) -> Any:

        output = [1] * self.fixed_actions

        assert (
            len(variable_list) == self.variable_actions
        ), f"{len(variable_list)}, {self.variable_actions}"

        output.extend([1 if i is not None else 0 for i in variable_list])

        return numpy.array(output, dtype=numpy.float64)


class MaxTimeExceededCondition(BaseCallable):
    def __init__(self, max_time: float, **kwargs) -> None:
        self.max_time = max_time
        super().__init__()

    def __call__(self, simulator: SimpleWorld, **kwargs) -> bool:
        if simulator.time >= self.max_time:
            return True
        return False


class EnemyEnteredBaseCondition(BaseCallable):
    def __call__(self, simulator: SimpleWorld) -> bool:
        agent = simulator.get_all_of_type(Types.AGENT)[0]

        for coll in simulator.get_collision_events():
            if "base" not in coll.names:
                continue
            elif agent in coll.names:
                continue
            return True
        return False


class AgentInterception(BaseCallable):
    def __call__(
        self, agent: Particle, simulator: SimpleWorld, base: Particle, **kwargs
    ) -> bool:
        return any(
            [
                agent.name in task.names and base.name not in task.names
                for task in simulator.get_collision_events()
            ]
        )


class AgentTaskCompleteCondition(BaseCallable):
    def __call__(
        self, agent: Particle, simulator: SimpleWorld, **kwargs
    ) -> bool:
        return any(
            [
                agent.name in task.names
                for task in simulator.get_untasked_agents()
            ]
        )


class ParticleAddedCondition(BaseCallable):
    def __call__(self, simulator: SimpleWorld, **kwargs) -> bool:

        return len(simulator.get_added_particles()) > 0


class AgentInterceptionReward(BaseCallable):
    def __init__(self, weight: float, **kwargs) -> None:
        self.weight = weight
        self.condition = AgentInterception()

    def __call__(
        self, agent: Particle, simulator: SimpleWorld, base: Particle, **kwargs
    ) -> float:

        if self.condition(agent=agent, simulator=simulator, base=base):
            return self.weight

        return 0


class EnemyEnteredBaseReward(BaseCallable):
    def __init__(self, weight: float, **kwargs):
        self.weight = weight
        self.condition = EnemyEnteredBaseCondition()

    def __call__(self, simulator: SimpleWorld, **kwargs) -> float:

        if self.condition(simulator):
            return self.weight

        return 0


class ZoneDefense(MultiAgentEnv):
    """
    Gym environment where agents must
    defend their zone from 'attackers'
    who try to attempt to enter the zone.

    Consists of at least one agent, and
    at least one attacker over the course
    of the episode.

    Agents are able to track down attackers
    with the ability to move faster than them.

    Agent actions consist of selecting one
    of the many attackers, with an option
    of 'none'

    The goal here is for agents to learn
    who is best suited when to take out
    an attacker by chasing them down,
    or when to remain at base, as an attempt
    to efficiently 'assign' entities

    TODO:
    1. Make sure the agent can actually reach the enemy
    2. Create action mask for valid choices
    """

    def __init__(self, env_config: dict):

        self._logger = logging.getLogger()
        self._setup_init(**env_config)

    def _setup_init(
        self,
        simultaneous_enemies: int,
        enemy_speed: float,
        agent_speed: float,
        grid_size: int,
        base_radius: float,
        agent_radius: float,
        enemy_radius: float,
        sim_step_size: float,
        logging_level: str = logging.DEBUG,
        total_episode_time: float = 120,
        reward_weights: dict = None,
        **kwargs,
    ):

        if reward_weights is None:
            reward_weights = {}

        # how many enemies can be in the sim at once
        self.simultaneous_enemies: int = simultaneous_enemies
        # total enemies to spawn over an episode (seconds)
        self.enemy_speed: float = enemy_speed
        self.agent_speed: float = agent_speed
        # how large the arena (assumed square)
        self.grid_size: int = grid_size
        self.base_radius: float = base_radius
        self.agent_radius: float = agent_radius
        self.enemy_radius: float = enemy_radius

        # create the simulation
        self.simulator = SimpleWorld(sim_step_size)
        # configure the logger
        self._logger.setLevel(logging_level)

        self._done_funcs = [
            MaxTimeExceededCondition(total_episode_time),
            EnemyEnteredBaseCondition(),
        ]

        self._query_funcs = [
            AgentTaskCompleteCondition(),
            AgentInterception(),
            ParticleAddedCondition(),
        ]

        self._reward_funcs = [
            AgentInterceptionReward(
                reward_weights[AgentInterceptionReward.__name__]
            ),
            EnemyEnteredBaseReward(
                reward_weights[EnemyEnteredBaseReward.__name__]
            ),
        ]

        self._observations: OrderedDict = self._build_observation_space(
            **kwargs
        )

    def _build_observation_space(self, **kwargs):

        obs: OrderedDict = OrderedDict()
        obs["agent_distance"] = DistanceToPoint(modifier=NormalizeBox())
        obs["agent_bearing"] = AbsoluteBearing()
        obs["agent_speed"] = Speed(modifier=NormalizeBox())

        for i in range(self.simultaneous_enemies):
            obs[f"enemy_{i}_distance"] = DistanceToPoint(
                modifier=NormalizeBox()
            )
            obs[f"enemy_{i}_bearing"] = AbsoluteBearing()
            obs[f"enemy_{i}_speed"] = Speed(modifier=NormalizeBox())

        obs["action_mask"] = ValidActions(
            fixed=1, variable=self.simultaneous_enemies
        )

        return obs

    @property
    def observation_space(self) -> spaces.Space:

        return spaces.Dict(
            OrderedDict(
                {key: obs.space() for key, obs in self._observations.items()}
            )
        )

    @property
    def action_space(self) -> spaces.Space:

        # index 0 = remain at base
        # index 1 - (max_enemies -1) = pick target
        return spaces.Discrete(self.simultaneous_enemies + 1)

    def reset(self) -> Any:

        # remove all entities and set to 0 enemies
        self.simulator.reset()

        # create the central base that enemies try to get to
        self.base = self.simulator.create_particle(
            name="base", type=Types.BASE
        )
        self.base.set_position(0, 0, 0)
        self.base.set_radius(self.base_radius)

        # add script to generate enemies
        def create_enemy(particle: Particle) -> None:
            start_x: float = numpy.random.uniform(0, self.grid_size)
            particle.set_position(start_x, self.grid_size, 0)
            particle.set_radius(self.enemy_radius)

            behavior = GoToPoint2D(
                end=self.base.position,
                speed=self.enemy_speed,
            )
            particle.add_behavior(behavior)

        enemy_script: Script = CreateEntityInterval(
            prefix="enemy",
            type=Types.ENEMY,
            interval=10,
            max=self.simultaneous_enemies,
            setup_func=create_enemy,
        )

        self.simulator.add_script(enemy_script)

        # create agent(s)
        self.agent = self.simulator.create_particle(
            name="agent", type=Types.AGENT
        )
        self.agent.set_position(0, 0, 0)
        self.agent.set_radius(self.agent_radius)

        return self._get_observation()

    def _get_observation(self) -> dict:

        agent_obs = OrderedDict()

        obs_def = self._observations

        agent_obs["agent_distance"] = obs_def["agent_distance"](
            name=self.agent.name,
            other=self.base.name,
            simulator=self.simulator,
        )
        agent_obs["agent_bearing"] = obs_def["agent_bearing"](
            name=self.agent.name,
            other=self.base.name,
            simulator=self.simulator,
        )
        agent_obs["agent_speed"] = obs_def["agent_speed"](
            name=self.agent.name,
            other=self.base.name,
            simulator=self.simulator,
        )

        enemies = order_enemies_by_distance(self.simulator, self.base)
        enemies_padded = enemies + [None] * (
            self.simultaneous_enemies - len(enemies)
        )

        for i, name in enumerate(enemies_padded):

            agent_obs[f"enemy_{i}_distance"] = obs_def[f"enemy_{i}_distance"](
                name=name, other=self.base.name, simulator=self.simulator
            )
            agent_obs[f"enemy_{i}_bearing"] = obs_def[f"enemy_{i}_bearing"](
                name=name, other=self.base.name, simulator=self.simulator
            )
            agent_obs[f"enemy_{i}_speed"] = obs_def[f"enemy_{i}_speed"](
                name=name, other=self.base.name, simulator=self.simulator
            )

        agent_obs["action_mask"] = obs_def["action_mask"](enemies_padded)

        return {self.agent.name: agent_obs}

    def _implement_actions(self, actions: Dict) -> None:

        for name in actions:
            self._logger.debug(f"agent picked {actions[name]}")
            if actions[name] == 0:

                at_base = calculations.in_bounds_of(self.agent, self.base)

                if at_base:
                    self.agent.add_behavior(RemainInLocationSeconds(5.0))
                    self._logger.info("Remaining at base")
                else:
                    self.agent.add_behavior(
                        GoToPoint2D(
                            end=self.base.position,
                            speed=self.agent_speed,
                        )
                    )
                    self._logger.info("Going back to base")

                continue

            enemies: List = order_enemies_by_distance(
                self.simulator, self.base
            )

            # subtract one b/c of go to base as 0
            target = enemies[actions[name] - 1]

            # calculate the intercept position
            intercept_pos = calculations.create_intercept_location(
                self.agent,
                self.simulator.get(target),
                self.agent_speed,
                self.enemy_speed,
            )

            self.agent.add_behavior(
                GoToPoint2D(
                    end=intercept_pos,
                    speed=self.agent_speed,
                )
            )
            self._logger.info(f"Intercepting {target}")

    def _time_to_query(self) -> bool:

        conditions: dict = {
            cond.__name__: cond(
                agent=self.agent, simulator=self.simulator, base=self.base
            )
            for cond in self._query_funcs
        }

        for c, value in conditions.items():
            if value:
                self._logger.debug(f"Condition {c} met")

        return any(conditions.values())

    def _remove_collided_enemies(self):

        for collision in self.simulator.get_collision_events():
            if self.base.name in collision.names:
                continue

            if self.agent.name not in collision.names:
                continue

            names = set(collision.names)
            names.remove(self.agent.name)
            enemy_to_remove = names.pop()

            self.simulator.remove_object(enemy_to_remove)

            self._logger.info(f"removed {enemy_to_remove}")

    def step(
        self, actions: Dict
    ) -> Tuple[Dict[str, Any], dict, float, Dict[str, Any]]:

        self._implement_actions(actions)
        self.simulator.update()

        while not self._time_to_query():
            self.simulator.update()

        rewards: dict = {self.agent.name: 0}
        dones: dict = {self.agent.name: False, "__all__": False}

        # collecting rewards
        rewards[self.agent.name] = sum(
            [
                reward(
                    agent=self.agent, simulator=self.simulator, base=self.base
                )
                for reward in self._reward_funcs
            ]
        )
        # collect dones
        done_conditions = [
            (dones.__class__.__name__, dones(self.simulator))
            for dones in self._done_funcs
        ]
        dones[self.agent.name] = any([i[1] for i in done_conditions])
        dones["__all__"] = dones[self.agent.name]

        # clean up sim for next round
        self._remove_collided_enemies()

        return self._get_observation(), rewards, dones, {}

    def render(self):

        from gym.envs.classic_control import rendering

        if self._display is None:
            self._display = rendering.Viewer(self.grid_size, self.grid_size)

            self.base_render = rendering.make_circle(radius=self.base.radius)
            base_trans = rendering.Transform()
            self.base_render.add_attr(base_trans)

            self._display.add_geom(self.base_render)

            self.agent_render = rendering.make_circle(radius=self.agent.radius)
            agent_trans = rendering.Transform()
            self.agent_render.add_attr(agent_trans)

            self._display.add_geom(self.agent_render)

        return self._display.render(True)
