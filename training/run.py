import ray
import os
import yaml
import datetime
import time
import math
import argparse
import importlib
import json
import models

from ray.rllib.agents.ppo import PPOTrainer

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


def load_yaml_from_file(path: str) -> dict:
    """Load the yaml file given the path

    :param path: path to yaml file
    :type path: str
    :return: dictionary of config
    :rtype: dict
    """

    with open(path) as f:
        config = yaml.load(f, yaml.SafeLoader)

    return config


def load_json_from_file(path: str) -> dict:
    """Loads the JSON from file path

    :param path: path to file
    :type path: str
    :return: dictionary of config
    :rtype: dict
    """
    with open(path) as f:
        config = json.load(f)

    return config


def import_class(path: str) -> type:
    """Returns the class definition given the path to class

    :param path: path to class in form "a.b.class"
    :type path: str
    :return: the class definition (not initialized)
    :rtype: [type]
    """
    module_path, class_name = path.rsplit(".", 1)

    return getattr(importlib.import_module(module_path), class_name)


def create_save_path(directory: str) -> str:
    """Creates a save path in the given directory

    :param directory: parent directory
    :type directory: str
    :return: string of the new path
    :rtype: str
    """

    day = datetime.date.today()
    time_now = math.floor(time.time())
    formatted_day = f"{day.year}-{day.month}-{day.day}-{time_now}"

    full_path: str = os.path.join(directory, formatted_day)

    os.makedirs(full_path)

    return full_path


def train(train_config: dict, env_config: dict):

    env_def = import_class(env_config[ENV_ENTRY])

    ray.init()
    trainer = PPOTrainer(
        env=env_def,
        config={
            **train_config,
            ENVIRONMENT_CONFIG: env_config,
        },
    )
    for _ in range(20):

        print(trainer.train())
        trainer.save()


def evaluate(save_path: str, checkpoint: str):

    config: dict = load_json_from_file(os.path.join(save_path, "params.json"))
    config["num_workers"] = 0
    config[ENVIRONMENT_CONFIG]["logging_level"] = "DEBUG"

    # TODO remove environment config to be left with trainer params
    ray.init()
    env_def: type = import_class(config[ENVIRONMENT_CONFIG][ENV_ENTRY])

    trainer = PPOTrainer(env=env_def, config=config)
    trainer.restore(checkpoint)

    env = env_def(config[ENVIRONMENT_CONFIG])

    done = False
    obs = env.reset()

    while not done:
        # env.render()
        actions = {
            agent: trainer.compute_single_action(obs[agent]) for agent in obs
        }
        obs, reward, dones, info = env.step(actions)
        done = dones["__all__"]


if __name__ == "__main__":

    args = train_parse_args()

    if args.eval:
        evaluate(save_path=args.load, checkpoint=args.checkpoint)
        exit(0)

    env_yaml = load_yaml_from_file(args.env_config)
    train_yaml = load_yaml_from_file(args.train_config)

    train(
        env_config=env_yaml,
        train_config=train_yaml,
    )
