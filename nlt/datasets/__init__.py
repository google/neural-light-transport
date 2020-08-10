from importlib import import_module


def get_dataset_class(name):
    mod = import_module('datasets.' + name)
    return mod.Dataset
