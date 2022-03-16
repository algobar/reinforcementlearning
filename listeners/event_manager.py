"""Module for listeners dispatched to wait for events"""


from dataclasses import dataclass, field
from typing import Dict, List
import typing
from listeners.cache import EventCache
from listeners.listeners import Listener

from simulation.messages import Message, SimulationState


@dataclass
class EventManager:
    """Handles Event Listeners"""

    cache: EventCache
    listeners_by_message_type: Dict[str, List[Listener]] = field(
        default_factory=dict
    )

    def add_listener(self, listener: Listener):
        """Registers Listeners and tags them for certain message types"""

        for msg_type in listener.message_types:

            if msg_type.__name__ not in self.listeners_by_message_type:
                self.listeners_by_message_type[msg_type.__name__] = []

            self.listeners_by_message_type[msg_type.__name__].append(listener)

    def notify(self, message: Message):
        """Notifies the listeners registered to the message type"""
        for listener in self.listeners_by_message_type[
            message.__class__.__name__
        ]:

            listener.notify(message, self.cache)

    def get_state(self) -> SimulationState:
        """Get the state of the sim"""
        return self.cache.get_messages(self.cache.STATE_MESSAGE)

    def get_messages_for(
        self, name: str, type: typing.Type = None
    ) -> List[Message]:
        """Queries the cache for messages by name"""
        return self.cache.get_messages(name, type)
