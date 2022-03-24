"""Module for listeners dispatched to wait for events"""


from dataclasses import dataclass, field
from typing import Callable, List
import typing
from listeners.cache import EventCache

from simulation.messages import Message, SimulationState


@dataclass
class EventManager:
    """Handles Event Listeners"""

    cache: EventCache
    listeners: List[Callable] = field(default_factory=list)

    def add_listener(self, listener: Callable):
        """Registers Listeners and tags them for certain message types"""

        self.listeners.append(listener)

    def notify(self, message: Message):
        """Notifies the listeners registered to the message type"""
        for listener in self.listeners:
            listener(message, self.cache)

    def get_state(self) -> SimulationState:
        """Get the state of the sim"""
        return self.cache.get_messages(self.cache.STATE_MESSAGE)

    def get_messages_for(
        self, name: str, type: typing.Type = None
    ) -> List[Message]:
        """Queries the cache for messages by name"""
        return self.cache.get_messages(name, type)
