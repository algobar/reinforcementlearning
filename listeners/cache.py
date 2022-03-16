from dataclasses import dataclass, field
from pyexpat.errors import messages
from typing import Dict, List
import typing

from simulation.messages import Message

STATE: str = "state"


@dataclass
class MessageCache:
    """Stores messages by the given message type"""

    messages: Dict[str, List[Message]] = field(default_factory=dict)

    def add_message(self, message: Message):

        if message.__class__.__name__ not in messages:
            self.messages[message.__class__.__name__] = []

        self.messages[message.__class__.__name__].append(message)

    def get_by_type(self, type: typing.Type):
        """Get the messages by the specified type"""
        return self.messages[type.__name__]

    def get_all_messages(self):
        """Convert message by types to single list"""
        output = []

        for messages in self.messages.values():
            output.extend(messages)

        return output


@dataclass
class EventCache:
    """Used to tie objects/names/ids to messages"""

    STATE_MESSAGE: str = STATE

    _cache: Dict[str, MessageCache] = field(default_factory=dict)

    def get_messages(
        self, name: str, type: typing.Type = None
    ) -> List[Message]:
        """Gets the messages for the given name"""

        if type:
            return self._cache[name].get_by_type(type)
        return self._cache[name].get_all_messages()

    def add_message(self, name: str, message: Message):
        """Record a new message for the name"""
        if name not in self._cache:
            self._cache[name] = []

        self._cache[name].append(message)

    def clear(self):
        """Clears all the names and data from cache"""
        self._cache.clear()
