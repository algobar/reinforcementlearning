"""Module to describe tasks, which are input into a simulation"""
from typing import List
from easyai.environments.types import Task


def create_task_list(*tasks) -> List[Task]:
    """Create a task list from variable number of tasks"""
    return [*tasks]


class LocationCartesian(Task):
    """Represents a cartesian coordinate"""

    x: float
    y: float
    z: float
    name: str = "LocationCartesian"


class Speed(Task):
    """Represents a single speed value"""

    speed: float
    name: str = "Speed"
