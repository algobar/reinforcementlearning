"""Module to describe tasks, which are input into a simulation

1. Need func to parse action dict to task List


"""
from typing import Any, Dict, List
from aiassembly.environments.types import Task


def create_task_list(*tasks) -> List[Task]:
    """Create a task list from variable number of tasks"""
    return [*tasks]


def build_task_list(actions: Dict[str, Any]) -> List[Task]:
    """Map actions from a dictionary into a task request list"""
    ...