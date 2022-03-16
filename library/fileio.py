import yaml
import json
import os
import datetime
import time
import math


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


def today_and_now_folder(directory: str) -> str:
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
