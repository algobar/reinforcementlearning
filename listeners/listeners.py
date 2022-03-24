from functools import wraps
from listeners.cache import EventCache, STATE
from simulation.messages import (
    Message,
    SimulationState,
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


@register_message(Collision)
def notify_collision(message: Collision, cache: EventCache, **kwargs):

    for particle in message.particles:
        cache.add_message(particle.name, message)


@register_message(SimulationState)
def notify_simulation_state(
    message: SimulationState, cache: EventCache, **kwargs
):

    cache.clear()
    cache.add_message(STATE, message)


@register_message(TaskComplete)
def notify_task_complete(message: TaskComplete, cache: EventCache, **kwargs):

    cache.add_message(message.particle.name, message)
