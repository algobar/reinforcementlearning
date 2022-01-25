import ray
from ray.rllib.agents.ppo import PPOTrainer
from environments.defend import ZoneDefense
import yaml
import models


def train(env_config: dict):

    from ray.rllib.models import ModelCatalog
    from models.variable_action import VariableActionModel

    ModelCatalog.register_custom_model(
        VariableActionModel.__name__, VariableActionModel
    )

    ray.init(local_mode=True)
    trainer = PPOTrainer(
        env=ZoneDefense,
        config={
            "env_config": env_config,
            "framework": "torch",
            "observation_filter": "NoFilter",
            "num_workers": 0,
            "model": {"custom_model": "VariableActionModel"},
        },
    )
    for i in range(10):

        print(trainer.train())


if __name__ == "__main__":

    with open(
        "/home/bob/reinforcementlearning/environments/config/defend_config.yaml"
    ) as f:
        env_config = yaml.load(f, yaml.SafeLoader)

    train(env_config=env_config)
