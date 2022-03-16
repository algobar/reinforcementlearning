from .base import BaseCallable, Collector
from simulation.particles import Types, Particle

from typing import List, Tuple, Dict
from simulation.messages import (
    Collision,
    Message,
    SimulationState,
    TaskComplete,
)

from listeners.event_manager import EventManager


class ConditionCollector(Collector):
    def __call__(self, **kwargs) -> Tuple[bool, Dict[str, Message]]:

        results = {cond.__name__: cond(**kwargs) for cond in self._callables}

        return any([res[0] for res in results]), results


class MaxTimeExceededCondition(BaseCallable):
    def __init__(self, max_time: float, **kwargs) -> None:
        self.max_time = max_time
        super().__init__()

    def __call__(
        self, event_manager: EventManager, **kwargs
    ) -> Tuple[bool, Message]:

        sim_state: SimulationState = event_manager.get_state()

        return sim_state.time > self.max_time, None


class EnemyEnteredBaseCondition(BaseCallable):
    def __call__(
        self, event_manager: EventManager, base: Particle, **kwargs
    ) -> Tuple[bool, Collision]:

        collisions: List[Collision] = event_manager.get_messages_for(
            base.name, Collision
        )

        for each_collision in collisions:

            if base not in each_collision.particles:

                continue

            types = {part.type for part in each_collision.particles}

            if Types.ENEMY not in types:

                continue

            return True, each_collision

        return False, None


class AgentInterception(BaseCallable):
    """Looks for agent interception of objects

    :param BaseCallable: [description]
    :type BaseCallable: [type]
    """

    def __call__(
        self, event_manager: EventManager, agent: Particle, **kwargs
    ) -> Tuple[bool, Message]:
        """Returns true if an agent has intercepted a type enemy"""

        collisions: List[Collision] = event_manager.get_messages_for(
            agent.name, Collision
        )

        for each_collision in collisions:

            types = {part.type for part in each_collision.particles}

            if Types.BASE in types:
                continue

            if all([Types.AGENT == part_type for part_type in types]):
                continue

            return True, each_collision

        return False, None


class AgentTaskCompleteCondition(BaseCallable):
    def __call__(
        self, agent: Particle, event_manager: EventManager, **kwargs
    ) -> bool:

        completed_tasks: List[TaskComplete] = event_manager.get_messages_for(
            agent.name, TaskComplete
        )

        if len(completed_tasks) < 1:
            return False, None

        if len(completed_tasks) > 1:
            raise ValueError("only expecting at most one completed task!")

        return True, completed_tasks[0]
