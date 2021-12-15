import yaml
from yaml.loader import SafeLoader


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_configs():
    return _load_config_yaml("./resources/train_config.yaml")


def _load_config_yaml(config_file):
    config_dict = yaml.load(open(config_file, 'rb'), SafeLoader)
    return Config(**config_dict)
