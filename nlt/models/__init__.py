from importlib import import_module


def get_model_class(name):
    mod = import_module('models.' + name)
    return mod.Model
