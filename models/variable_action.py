from typing import List, Dict, Tuple
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.typing import ModelConfigDict
from torch import TensorType, nn
import gym
import numpy as np


class VariableActionModel(FullyConnectedNetwork):
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:

        output, state = super().forward(input_dict, state, seq_lens)

        # mask the action outputs]
        mask = None

        return output, state
