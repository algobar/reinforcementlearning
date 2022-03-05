from .base import BaseCallable, Collector
from simulation.simulator import SimpleWorld
from simulation.particles import Types, Particle

from typing import Tuple, Dict, Any


class ConditionCollector(Collector):
    def __call__(self, **kwargs) -> Tuple[bool, Dict[str, bool]]:

        results = {cond.__name__: cond(**kwargs) for cond in self._callables}

        return any(results.values()), results


class MaxTimeExceededCondition(BaseCallable):
    def __init__(self, max_time: float, **kwargs) -> None:
        self.max_time = max_time
        super().__init__()

    def __call__(self, simulator: SimpleWorld, **kwargs) -> bool:
        if simulator.time >= self.max_time:
            return True
        return False


class EnemyEnteredBaseCondition(BaseCallable):
    def __call__(self, simulator: SimpleWorld, **kwargs) -> bool:
        all_agent = simulator.get_all_of_type(Types.AGENT)

        for coll in simulator.get_collision_events():
            if "base" not in coll.names:
                continue
            elif any([agent in coll.names for agent in all_agent]):
                continue
            return True
        return False


class AgentInterception(BaseCallable):
    """Looks for agent interception of objects

    :param BaseCallable: [description]
    :type BaseCallable: [type]
    """

    def __call__(
        self, agent: Particle, simulator: SimpleWorld, base: Particle, **kwargs
    ) -> bool:
        """Returns true if an agent has intercepted a type enemy

        :param agent: [description]
        :type agent: Particle
        :param simulator: [description]
        :type simulator: SimpleWorld
        :param base: [description]
        :type base: Particle
        :return: [description]
        :rtype: bool
        """

        set_enemies = set(simulator.get_all_of_type(Types.ENEMY))

        for collision in simulator.get_collision_events():
            coll_set = set(collision.names)
            # an enemy must be in the set
            if len(set_enemies.intersection(coll_set)) == 0:
                continue
            # agent must be in the set
            elif not agent.name in coll_set:
                continue
            return True
        return False


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
