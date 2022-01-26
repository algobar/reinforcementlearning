from operator import mod
import ray
import os
import yaml
import datetime
import argparse
import importlib

from ray.rllib.agents.ppo import PPOTrainer

ENVIRONMENT_CONFIG: str = "env_config"
ENV_ENTRY: str = "env"


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", type=str)
    parser.add_argument("--train-config", type=str)
    parser.add_argument("--save", type=str)

    return parser.parse_args()


def load_yaml(path: str) -> dict:
    """Load the yaml file given the path

    :param path: path to yaml file
    :type path: str
    :return: dictionary of config
    :rtype: dict
    """

    with open(path) as f:
        config = yaml.load(f, yaml.SafeLoader)

    return config


def import_class(path: str):

    module, class_name = path.rsplit(".", 1)

    return importlib.import_module(module, class_name)


def create_save_path(directory: str) -> str:
    """Creates a save path in the given directory

    :param directory: parent directory
    :type directory: str
    :return: string of the new path
    :rtype: str
    """

    day = datetime.date()
    time = day.timetuple()
    formatted_day = f"{day.year}-{day.month}-{day.day}-\
        {time.tm_hour}{time.tm_min}{time.tm_sec}"

    full_path: str = os.path.join(directory, formatted_day)

    os.makedirs(full_path)

    return full_path


def train(train_config: dict, env_config: dict, save_path: str):

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
        trainer.save(save_path)


if __name__ == "__main__":

    args = parse_args()

    save_path = create_save_path(args.save)

    train(
        env_config=args.env_config,
        train_config=args.train_config,
        save_path=save_path,
    )
