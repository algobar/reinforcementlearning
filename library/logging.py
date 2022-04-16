import json
import logging
from logging import FileHandler, LogRecord
from logging import Formatter


class JSONFormatter(Formatter):
    def format(self, record: LogRecord) -> str:

        msg = record.msg

        return json.dumps(msg)


def create_stdout_logger(name: str, level=logging.DEBUG) -> logging.Logger:
    """Create a generic stdout logger with unique name and level if specified"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

    return logger


def create_json_logger(
    name: str, filepath: str, level=logging.DEBUG
) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    handler = FileHandler(filepath, mode="w")
    handler.setLevel(level)
    handler.setFormatter(JSONFormatter())

    logger.addHandler(handler)

    return logger
