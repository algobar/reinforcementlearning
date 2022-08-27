import importlib


def import_class(path: str) -> type:
    """Returns the class definition given the path to class

    :param path: path to class in form "a.b.class"
    :type path: str
    :return: the class definition (not initialized)
    :rtype: [type]
    """
    module_path, class_name = path.rsplit(".", 1)

    return getattr(importlib.import_module(module_path), class_name)
