from typing import Any, Callable, Dict, List, Tuple
from aiassembly.environments.types import StoredStateInfo

QueryStatus = bool
DoneStatus = bool

NamedQueryStatus = Tuple[str, QueryStatus]
RepeatedQueryStatus = List[NamedQueryStatus]

NamedQueryStatus = Dict[str, RepeatedQueryStatus]
AgentDone = Dict[str, bool]
AgentReward = Dict[str, float]

InfoDict = Dict[str, float]

QueryAgentFunc = Callable[[StoredStateInfo], Tuple[QueryStatus, NamedQueryStatus]]
FeatureProcessFunc = Callable[[StoredStateInfo, NamedQueryStatus], Tuple[Any, InfoDict]]
DoneProcessFunc = Callable[[StoredStateInfo, NamedQueryStatus], Tuple[AgentDone, InfoDict]]
RewardProcessFunc = Callable[[StoredStateInfo, NamedQueryStatus], Tuple[AgentReward, InfoDict]]