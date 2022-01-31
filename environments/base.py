from abc import ABC
from typing import Any, Tuple, Dict


class BaseCallable(ABC):
    def __init__(self, **kwargs) -> None:
        self.__name__ = self.__class__.__name__

    def __call__(self, **kwargs) -> Any:
        ...


class Collector(BaseCallable):
    """Used to query base callables and summarize their outputs"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._callables = []

    def append(self, callable: BaseCallable):

        self._callables.append(callable)

    def __call__(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        return super().__call__(**kwargs)
