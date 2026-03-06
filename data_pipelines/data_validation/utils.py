import json

import yaml


def load_json(path):
    with open(path, 'r') as openfile:
        json_object = json.load(openfile)
    return json_object


def load_yaml(path):
    with open(path, 'r') as yml_file:
        data = yaml.load(yml_file, Loader=yaml.FullLoader)
    return data
