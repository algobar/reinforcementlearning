from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict
import socketio
import json
import time


@dataclass
class RenderData:
    """Defines acceptable render data to pass"""

    name: str
    x: float
    y: float
    radius: float

    def __post_init__(self):

        self.x = float(self.x)
        self.y = float(self.y)

    def to_json(self) -> str:
        """Returns object as json"""
        render_dict = asdict(self)
        return render_dict


class Render(ABC):
    """Generic Base Class for passing redering information to"""

    @abstractmethod
    def render(self, sim_data: Dict[str, RenderData], **kwds) -> Any:
        """Renders the data given the simulation information"""
        ...


class SocketIORender(Render):
    """Passes data to socketio server"""

    def __init__(self, address: str) -> None:
        self.sio = socketio.Client()
        try:
            self.sio.connect(address)
        except socketio.client.exceptions.BadNamespaceError as bad_host:
            raise RuntimeError(
                "failure to connect to socketio server"
            ) from bad_host

        super().__init__()

    def render(self, sim_data: Dict[str, RenderData], **kwds) -> None:
        """Converts the sim data to JSON and emits to the server"""
        as_json = {key: value.to_json() for key, value in sim_data.items()}
        self.sio.emit("sim", data=as_json)
