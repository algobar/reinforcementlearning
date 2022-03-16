from abc import ABC, abstractmethod
from functools import wraps
from listeners.cache import EventCache, STATE
from simulation.messages import (
    Message,
    SimulationState,
    SimulationStep,
    Collision,
    TaskComplete,
)


def register_message(message_type: Message):
    """Given the

    :param message_type: valid message type
    :type message_type: Message
    """

    def message_decorator(func):
        @wraps(func)
        def wrapper(message: Message, *args, **kwargs):
            if message.__class__.__name__ != message_type.__name__:
                return None
            return func(message, *args, **kwargs)

        return wrapper

    return message_decorator


class Listener(ABC):
    @abstractmethod
    def notify(message: Message, cache, **kwargs):
        """Gives the listener the message to handle and a cache to update"""
        ...


class CacheResetListener(Listener):
    """Job is to look for simulation step messages
    and reset the cache for any new events"""

    @staticmethod
    @register_message(SimulationStep)
    def notify(message: SimulationStep, cache: EventCache, **kwargs):

        cache.clear()


class CollisionListener(Listener):
    @staticmethod
    @register_message(Collision)
    def notify(message: Collision, cache: EventCache, **kwargs):

        for particle in message.particles:
            cache.add_message(particle.name, message)


class SimulationStateListener(Listener):
    @staticmethod
    @register_message(SimulationState)
    def notify(message: SimulationState, cache: EventCache, **kwargs):

        cache.add_message(STATE, message)


class TaskComplete(Listener):
    @staticmethod
    @register_message(TaskComplete)
    def notify(message: TaskComplete, cache: EventCache, **kwargs):

        cache.add_message(message.particle.name, message)
