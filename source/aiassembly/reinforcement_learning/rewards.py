from functools import reduce
import operator
import typing

Reward = float
RewardWeight = float


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
