import ray
import os
import argparse
import utils
import models
from ray.rllib.agents.ppo import PPOTrainer

from simulation.rendering import SocketIORender

from pprint import pprint

ENVIRONMENT_CONFIG: str = "env_config"
ENV_ENTRY: str = "env"


def train_parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", type=str)
    parser.add_argument("--train-config", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--load", type=str)
    parser.add_argument("--checkpoint", type=str)

    return parser.parse_args()


def train(train_config: dict, env_config: dict):
    """Trains agent given the training config file
    and the environment config file

    :param train_config: specifies trainer RLLIB params
    :type train_config: dict
    :param env_config: specifies env params
    :type env_config: dict
    """

    env_def = utils.import_class(env_config[ENV_ENTRY])

    ray.init()
    trainer = PPOTrainer(
        env=env_def,
        config={
            **train_config,
            ENVIRONMENT_CONFIG: env_config,
        },
    )
    for _ in range(60):

        print(trainer.train())
        trainer.save()


def evaluate(save_path: str, checkpoint: str):
    """Evaluates the environment given the path
    to the checkpoint folder, and path to the
    checkpoint file

    :param save_path: where the training run data was saved
    :type save_path: str
    :param checkpoint: checkpoint file
    :type checkpoint: str
    """

    config: dict = utils.load_json_from_file(
        os.path.join(save_path, "params.json")
    )
    config["num_workers"] = 0
    config[ENVIRONMENT_CONFIG]["logging_level"] = "DEBUG"

    # TODO remove environment config to be left with trainer params
    ray.init()
    env_def: type = utils.import_class(config[ENVIRONMENT_CONFIG][ENV_ENTRY])

    trainer = PPOTrainer(env=env_def, config=config)
    trainer.restore(checkpoint)

    env = env_def(config[ENVIRONMENT_CONFIG])

    obs = env.reset()

    env.simulator.add_render(SocketIORender("http://localhost:3000"))

    while True:
        actions = {
            agent: trainer.compute_single_action(obs[agent]) for agent in obs
        }

        obs, reward, dones, info = env.step(actions)
        done = dones["__all__"]

        if done:
            obs = env.reset()


if __name__ == "__main__":

    args = train_parse_args()

    if args.eval:
        evaluate(save_path=args.load, checkpoint=args.checkpoint)
        exit(0)

    env_yaml = utils.load_yaml_from_file(args.env_config)
    train_yaml = utils.load_yaml_from_file(args.train_config)

    train(
        env_config=env_yaml,
        train_config=train_yaml,
    )
