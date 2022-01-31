"""General python based utility functions"""

import os
import yaml
import datetime
import time
import math
import importlib
import json


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
