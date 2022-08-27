from functools import reduce
import operator
import typing

Reward = float
RewardWeight = float
RewardFuncType = typing.Callable[[object, object], float]


def apply_scaling(
    rewards: typing.List[Reward], scales: typing.List[RewardWeight]
) -> typing.List[float]:
    """Multiply each reward by its corresponding scaled value"""

    assert len(rewards) == len(scales), "length mismatch"

    return [r * s for r, s in zip(rewards, scales)]


def collect_rewards(
    rewards: typing.List[Reward],
    scales: typing.List[RewardWeight],
    operator=operator.add,
) -> float:
    """Given the rewards and their weights, scale
    the rewards and apply the operator to reduce to single value"""
    scaled = apply_scaling(rewards, scales)
    result = reduce(operator, scaled)
    return result


"""
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

        if self.condition(**kwargs)[0]:
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
"""
