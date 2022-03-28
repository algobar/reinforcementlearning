"""Module for listeners dispatched to wait for events"""


from dataclasses import dataclass, field
from logging import Logger
import logging
from typing import Callable, List
import typing
from listeners.cache import EventCache

from simulation.messages import Message, SimulationState


@dataclass
class EventManager:
    """Handles Event Listeners"""

    cache: EventCache
    listeners: List[Callable] = field(default_factory=list)
    logger: Logger = None

    def __post_init__(self):

        if self.logger:
            return

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(ch)

    def add_listener(self, listener: Callable):
        """Registers Listeners and tags them for certain message types"""

        self.listeners.append(listener)

    def notify(self, message: Message):
        """Notifies the listeners registered to the message type"""

        self.logger.info(
            f"manager notified of: {message.__class__.__name__} message"
        )
        for listener in self.listeners:
            listener(message, self.cache)

    def get_state(self) -> SimulationState:
        """Get the state of the sim"""

        return self.cache.get_messages(self.cache.STATE_MESSAGE)[-1]

    def get_messages_for(
        self, name: str, type: typing.Type = None
    ) -> List[Message]:
        """Queries the cache for messages by name"""
        return self.cache.get_messages(name, type)
