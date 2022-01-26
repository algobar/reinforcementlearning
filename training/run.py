import ray
import os
import yaml
import datetime
import time
import math
import argparse
import importlib
import models

from ray.rllib.agents.ppo import PPOTrainer

ENVIRONMENT_CONFIG: str = "env_config"
ENV_ENTRY: str = "env"


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", type=str)
    parser.add_argument("--train-config", type=str)
    parser.add_argument("--save", type=str)

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

    env_def = import_class(env_config.pop(ENV_ENTRY))

    ray.init()
    trainer = PPOTrainer(
        env=env_def,
        config={
            **train_config,
            ENVIRONMENT_CONFIG: env_config,
        },
    )
    for _ in range(10):

        print(trainer.train())
        trainer.save()


if __name__ == "__main__":

    args = parse_args()

    env_yaml = load_yaml_from_file(args.env_config)
    train_yaml = load_yaml_from_file(args.train_config)

    train(
        env_config=env_yaml,
        train_config=train_yaml,
    )
