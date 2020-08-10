def config2dict(config):
    """Assumes the configuration .ini has only the default section.
    """
    config_dict = {}
    for k, v in config.items('DEFAULT'):
        assert k not in config_dict, "Duplicate flags not allowed"
        config_dict[k] = v
    return config_dict
