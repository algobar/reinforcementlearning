import logging
from typing import Any, Dict, List, OrderedDict, Tuple
from gym import spaces
from ray.rllib.env import MultiAgentEnv
import numpy
from environments.base import build_collector_from_config
from environments.observations import (
    AbsoluteBearing,
    DistanceToPoint,
    Speed,
    ValidActions,
)
from environments.conditions import ConditionCollector
from environments.reward import RewardCollector
from listeners.cache import EventCache
from listeners.listeners import (
    notify_collision,
    notify_simulation_state,
    notify_task_complete,
)
from simulation.messages import Collision, SimulationState
from simulation.behaviors import (
    go_to_point_2d,
    remain_in_location_seconds,
)
from simulation.particles import (
    Particle,
    Types,
    create_particle,
)
from simulation.scripts import CreateEntityInterval, Script
from simulation import calculations
from simulation.simulator import Simulator

from ..modifier import NormalizeBox
from listeners.event_manager import EventManager


def order_enemies_by_distance(
    simulator: SimulationState, poi: Particle
) -> List:
    """Orders the enemies by distance to the given particle

    :param simulator: [description]
    :type simulator: SimpleWorld
    :param poi: [description]
    :type poi: Particle
    :return: [description]
    :rtype: List
    """
    enemy_distance = [
        (
            enemy,
            calculations.distance_between(simulator.get(enemy), poi),
        )
        for enemy in simulator.get_all_of_type(Types.ENEMY)
    ]

    sorted_enemies_by_distance = sorted(enemy_distance, key=lambda x: x[1])

    return [enemy[0] for enemy in sorted_enemies_by_distance]


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

        self._setup_init(**env_config)

    def _setup_init(
        self,
        number_agents: int,
        simultaneous_enemies: int,
        enemy_speed: float,
        agent_speed: float,
        grid_size: int,
        base_radius: float,
        agent_radius: float,
        enemy_radius: float,
        sim_step_size: float,
        enemy_creation_interval: float,
        dones: Dict,
        rewards: Dict,
        conditions: Dict,
        logging_level: str = logging.DEBUG,
        **kwargs,
    ):

        logging.basicConfig(level=logging_level)
        self._logger = logging.getLogger()

        self.number_agents = number_agents
        self.simultaneous_enemies: int = simultaneous_enemies
        # total enemies to spawn over an episode (seconds)
        self.enemy_speed: float = enemy_speed
        self.agent_speed: float = agent_speed
        # how large the arena (assumed square)
        self.grid_size: int = grid_size
        self.base_radius: float = base_radius
        self.agent_radius: float = agent_radius
        self.enemy_radius: float = enemy_radius
        self.enemy_creation_interval: float = enemy_creation_interval

        # create the simulation
        self.simulator = Simulator(sim_step_size)
        # create the event handler
        self.event_handler = EventManager(EventCache())

        self.event_handler.add_listener(notify_collision)
        self.event_handler.add_listener(notify_simulation_state)
        self.event_handler.add_listener(notify_task_complete)

        self._done_funcs = build_collector_from_config(
            ConditionCollector, dones
        )
        self._query_funcs = build_collector_from_config(
            ConditionCollector, conditions
        )
        self._reward_funcs = build_collector_from_config(
            RewardCollector, rewards
        )

        self._observations: OrderedDict = self._build_observation_space(
            **kwargs
        )

    def _build_observation_space(self, **kwargs):

        obs: OrderedDict = OrderedDict()

        for i in range(self.number_agents):

            obs[f"agent_{i}_distance"] = DistanceToPoint(
                modifier=NormalizeBox()
            )
            obs[f"agent_{i}_bearing"] = AbsoluteBearing()
            obs[f"agent_{i}_speed"] = Speed(modifier=NormalizeBox())

            obs[f"agent_{i}_to_agent"] = DistanceToPoint(
                modifier=NormalizeBox(), low=0, high=10
            )

            obs[f"agent_{i}_bearing_to_agent"] = AbsoluteBearing()

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
        """Returns the observation space"""
        return spaces.Dict(
            OrderedDict(
                {key: obs.space() for key, obs in self._observations.items()}
            )
        )

    @property
    def action_space(self) -> spaces.Space:
        """Defines the space as a selection of target to choose from, plus a go-home"""

        return spaces.Discrete(self.simultaneous_enemies + 1)

    def reset(self) -> Any:
        """Resets the env to starting point"""
        # remove all entities and set to 0 enemies
        self.simulator.reset()

        # create the central base that enemies try to get to
        self.base: Particle = create_particle(name="base", type=Types.BASE)
        self.base.set_position(self.grid_size / 2, self.grid_size / 2, 0)
        self.base.radius = self.base_radius
        self.simulator.add_particle(self.base)
        # add script to generate enemies

        def create_enemy(particle: Particle) -> None:

            start_angle_rad = numpy.random.uniform(0, 2 * numpy.pi)

            start_cos: float = numpy.cos(start_angle_rad)
            start_sin: float = numpy.sin(start_angle_rad)

            offset: float = self.grid_size / 2

            start_x = offset * start_cos + offset
            start_y = offset * start_sin + offset

            particle.set_position(start_x, start_y, 0)
            particle.radius = self.enemy_radius

            behavior = go_to_point_2d(
                end=self.base.position,
                speed=self.enemy_speed,
            )
            particle.behavior = behavior

        enemy_script: Script = CreateEntityInterval(
            prefix="enemy",
            type=Types.ENEMY,
            interval=self.enemy_creation_interval,
            max=self.simultaneous_enemies,
            setup_func=create_enemy,
        )

        self.simulator.register_script(enemy_script)

        # create agent(s)
        self.agents = {}
        for i in range(self.number_agents):
            agent: Particle = create_particle(
                name=f"agent_{i}", type=Types.AGENT
            )
            agent.set_position(
                self.base.position[0],
                self.base.position[1],
                self.base.position[2],
            )
            agent.radius = self.agent_radius
            self.agents[agent.name] = agent
            self.simulator.add_particle(agent)

        return self._get_observation([a for a in self.agents])

    def _get_observation(self, agents: List[str]) -> OrderedDict:

        return OrderedDict(
            {agent: self._get_observation_agent(agent) for agent in agents}
        )

    def _get_observation_agent(self, agent_name: str) -> dict:

        agent_self = self.agents[agent_name]
        agent_obs = OrderedDict()
        obs_def = self._observations

        for i, each_agent in enumerate(self.agents.values()):

            agent_obs[f"agent_{i}_distance"] = obs_def[f"agent_{i}_distance"](
                name=each_agent.name,
                other=self.base.name,
                state=self.event_handler.get_state(),
            )
            agent_obs[f"agent_{i}_bearing"] = obs_def[f"agent_{i}_bearing"](
                name=each_agent.name,
                other=self.base.name,
                state=self.event_handler.get_state(),
            )
            agent_obs[f"agent_{i}_speed"] = obs_def[f"agent_{i}_speed"](
                name=each_agent.name,
                other=self.base.name,
                state=self.event_handler.get_state(),
            )

            agent_obs[f"agent_{i}_to_agent"] = obs_def[f"agent_{i}_to_agent"](
                name=each_agent.name,
                other=agent_self.name,
                state=self.event_handler.get_state(),
            )

            agent_obs[f"agent_{i}_bearing_to_agent"] = obs_def[
                f"agent_{i}_bearing_to_agent"
            ](
                name=each_agent.name,
                other=agent_self.name,
                state=self.event_handler.get_state(),
            )

        enemies = order_enemies_by_distance(
            self.event_handler.get_state(), self.base
        )
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

        return agent_obs

    def _implement_actions_agent(self, actions: Dict) -> None:

        for name in actions:
            agent: Particle = self.agents[name]
            self._logger.info(f"{name} picked {actions[name]}")
            if actions[name] == 0:

                at_base = calculations.in_bounds_of(agent, self.base)

                if at_base:
                    agent.behavior = remain_in_location_seconds(5.0)
                    self._logger.info("Remaining at base")
                else:
                    agent.behavior = go_to_point_2d(
                        end=self.base.position,
                        speed=self.agent_speed,
                    )
                    self._logger.info("Going back to base")

                continue

            enemies: List = order_enemies_by_distance(
                self.simulator, self.base
            )

            # subtract one b/c of go to base as 0
            target = enemies[actions[name] - 1]

            # TODO add a None condition if fails!
            intercept_pos = calculations.create_intercept_location(
                agent,
                self.simulator.get(target),
                self.agent_speed,
                self.enemy_speed,
            )

            agent.behavior = go_to_point_2d(
                end=intercept_pos,
                speed=self.agent_speed,
            )
            self._logger.info(f"Intercepting {target}")

    def _time_to_query(self) -> Dict[str, bool]:

        query_dict: dict = {}

        for name, particle in self.agents.items():
            should_query, cond = self._query_funcs(
                agent=particle, simulator=self.simulator, base=self.base
            )
            query_dict[name] = should_query
        return query_dict

    def _remove_collided_enemies(self):
        """Removes enemies from scenario if collision occurs"""
        for agent in self.agents.values():
            for collision in self.event_handler.get_messages_for(
                agent.name, type=Collision
            ):
                if self.base.name in collision.names:
                    continue

                if all(["agent" in name for name in collision.names]):
                    continue

                names = set(collision.names)
                names.remove(agent.name)
                enemy_to_remove = names.pop()

                self.simulator.remove_particle(enemy_to_remove)

                self._logger.info(f"removed {enemy_to_remove}")

    def step(
        self, action_dict: Dict
    ) -> Tuple[Dict[str, Any], dict, float, Dict[str, Any]]:
        """Takes actions from policy and processes next obs, rwd, dones"""

        self._implement_actions_agent(action_dict)
        self.simulator.update()

        should_query: Dict[str, bool] = self._time_to_query()
        while not any(should_query.values()):
            self.simulator.update()
            should_query = self._time_to_query()

        agents_to_return = [
            agent for agent in should_query if should_query[agent]
        ]

        assert len(agents_to_return) > 0, f"{should_query}"

        self._logger.info(f"getting actions for {agents_to_return}")

        # get dones for agent to query
        dones: dict = {agent: False for agent in agents_to_return}
        dones["__all__"] = False

        for agent in agents_to_return:
            done, why = self._done_funcs(
                agent=self.agents[agent],
                base=self.base,
                simulator=self.simulator,
            )
            dones[agent] = done
            # mark all done if at least one done
            dones["__all__"] = done

        # collect everyone else's obs
        if dones["__all__"]:
            agents_to_return.extend(
                [
                    agent
                    for agent in self.agents
                    if agent not in agents_to_return
                ]
            )

        rewards: dict = {}
        # collecting rewards and dones
        for agent in agents_to_return:

            rwd, _ = self._reward_funcs(
                agent=self.agents[agent],
                base=self.base,
                simulator=self.simulator,
            )
            rewards[agent] = rwd

        # clean up sim for next round
        self._remove_collided_enemies()
        return self._get_observation(agents_to_return), rewards, dones, {}
