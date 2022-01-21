from abc import ABC
import logging
from typing import Any, Dict, List, OrderedDict, Tuple
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


class BaseCallable(ABC):
    def __init__(self, **kwargs) -> None:
        self.__name__ = self.__class__.__name__

    def __call__(self, **kwargs) -> Any:
        ...


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
    def __call__(self, agent: Particle, simulator: SimpleWorld, **kwargs) -> bool:
        return any(
            [agent.name in task.names for task in simulator.get_untasked_agents()]
        )


class ParticleAddedCondition(BaseCallable):
    def __call__(self, simulator: SimpleWorld, **kwargs) -> bool:

        return len(simulator.get_added_particles()) > 0


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
        **kwargs,
    ):

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

    @property
    def observation_space(self) -> spaces.Space:
        """Define the observation space as follows:

        - Distance to zone
        - True bearing to zone
        - For each enemy (up to max):
           - Distance to zone
           - True bearing to zone


        Returns:
            spaces.Space: [description]
        """
        distance = spaces.Box(low=numpy.array([-100]), high=numpy.array([100]))
        zone_bearing_x = spaces.Box(low=numpy.array([-1]), high=numpy.array([1]))
        zone_bearing_y = zone_bearing_x

        space: dict = {
            "agent_distance": distance,
            "agent_bearing_x": zone_bearing_x,
            "agent_bearing_y": zone_bearing_y,
        }

        for i in range(self.simultaneous_enemies):
            space[f"enemy_{i}_distance"] = distance
            space[f"enemy_{i}_bearing_x"] = zone_bearing_x
            space[f"enemy_{i}_bearing_y"] = zone_bearing_y

        space["avail_actions"] = spaces.Discrete(self.simultaneous_enemies + 1)

        return spaces.Dict(space)

    @property
    def action_space(self) -> spaces.Space:

        # index 0 = remain at base
        # index 1 - (max_enemies -1) = pick target
        return spaces.Discrete(self.simultaneous_enemies + 1)

    def reset(self) -> Any:

        # remove all entities and set to 0 enemies
        self.simulator.reset()

        # create the central base that enemies try to get to
        self.base = self.simulator.create_particle(name="base", type=Types.BASE)
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
        self.agent = self.simulator.create_particle(name="agent", type=Types.AGENT)
        self.agent.set_position(0, 0, 0)
        self.agent.set_radius(self.agent_radius)

        return self._get_observation()

    def _order_enemies_by_distance(self) -> List:

        enemy_distance = [
            (
                enemy,
                calculations.distance_between(self.simulator.get(enemy), self.base),
            )
            for enemy in self.simulator.get_all_of_type(Types.ENEMY)
        ]

        sorted_enemies_by_distance = sorted(enemy_distance, key=lambda x: x[1])

        return [enemy[0] for enemy in sorted_enemies_by_distance]

    def _get_observation(self) -> dict:

        obs: dict = {self.agent.name: None}

        # collect obs for now
        agent_obs = OrderedDict()

        distance_to_base = calculations.distance_between(self.agent, self.base)

        base_bearing: float = calculations.absolute_bearing_between(
            self.agent, self.base
        )

        agent_obs["agent_distance"] = distance_to_base
        agent_obs["agent_bearing_x"] = numpy.sin(base_bearing)
        agent_obs["agent_bearing_y"] = numpy.cos(base_bearing)

        enemy_by_distance = self._order_enemies_by_distance()

        for i in range(self.simultaneous_enemies):

            if len(enemy_by_distance) < i + 1:
                distance = 0
                bearing = 0
            else:
                enemy_to_observe = self.simulator.get(enemy_by_distance[i])
                distance = calculations.distance_between(
                    self.agent,
                    enemy_to_observe,
                )
                bearing = calculations.absolute_bearing_between(
                    self.agent,
                    enemy_to_observe,
                )

            agent_obs[f"enemy_{i}_distance"] = numpy.array(distance)
            agent_obs[f"enemy_{i}_bearing_x"] = numpy.array(numpy.sin(bearing))
            agent_obs[f"enemy_{i}_bearing_y"] = numpy.array(numpy.cos(bearing))

        obs[self.agent.name] = agent_obs

        return obs

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

            enemies: List = self._order_enemies_by_distance()

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

    def step(self, actions: Dict) -> Tuple[Dict[str, Any], dict, float, Dict[str, Any]]:

        self._implement_actions(actions)
        self.simulator.update()

        while not self._time_to_query():
            self.simulator.update()

        # init output containers
        obs = self._get_observation()
        rewards: dict = {self.agent.name: 0}
        dones: dict = {self.agent.name: False, "__all__": False}

        # collecting rewards

        # collect dones
        done_conditions = [
            (dones.__class__.__name__, dones(self.simulator))
            for dones in self._done_funcs
        ]
        dones[self.agent.name] = any([i[1] for i in done_conditions])
        dones["__all__"] = dones[self.agent.name]
        print(done_conditions)

        # clean up sim for next round
        self._remove_collided_enemies()

        return obs, rewards, dones, {}

    def render(self):

        ...
