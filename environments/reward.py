from typing import Tuple, Dict, Any
from .base import BaseCallable, Collector

from .conditions import (
    AgentInterception,
    EnemyEnteredBaseCondition,
)


class RewardCollector(Collector):
    def __call__(self, **kwargs) -> Tuple[float, Dict[str, float]]:

        results = {rwd.__name__: rwd(**kwargs) for rwd in self._callables}

        return sum(results.values()), results


class AgentInterceptionReward(BaseCallable):
    def __init__(self, weight: float, **kwargs) -> None:
        self.weight = weight
        self.condition = AgentInterception()
        super().__init__()

    def __call__(self, **kwargs) -> float:

        if self.condition(**kwargs):
            return self.weight

        return 0


class EnemyEnteredBaseReward(BaseCallable):
    def __init__(self, weight: float, **kwargs):
        self.weight = weight
        self.condition = EnemyEnteredBaseCondition()
        super().__init__()

    def __call__(self, **kwargs) -> float:

        if self.condition(**kwargs):
            return self.weight

        return 0
