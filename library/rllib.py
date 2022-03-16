import os


def rllib_format_checkpoint_folder(value: int) -> str:

    return f"checkpoint_{value:06d}"


def rllib_format_checkpoint_file(value: int) -> str:

    return f"checkpoint-{value}"


def rllib_get_checkpoint_path(parent: str, value: int) -> str:

    return os.path.join(
        parent,
        rllib_format_checkpoint_folder(value),
        rllib_format_checkpoint_file(value),
    )
