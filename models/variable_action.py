from typing import List, Dict, Tuple
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.typing import ModelConfigDict
from torch import TensorType
import torch


def mask_logits(logits: TensorType, mask: TensorType, device):
    """Mask the outputs of the network to be extremely small
    to prevent them from being chosen by the network"""

    return torch.where(mask < 1, torch.tensor(-1e8).to(device), logits)


class VariableActionModel(FullyConnectedNetwork):
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:

        output, state = super().forward(input_dict, state, seq_lens)

        mask = input_dict["obs"]["action_mask"]

        masked_output = mask_logits(output, mask, mask.device)

        return masked_output, state
