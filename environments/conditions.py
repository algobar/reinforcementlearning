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
