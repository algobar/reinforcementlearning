from ray.rllib.models import ModelCatalog
from models.variable_action import VariableActionModel

ModelCatalog.register_custom_model(
    VariableActionModel.__name__, VariableActionModel
)
