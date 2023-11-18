import os
import logging
import yaml


def check_file(file, mode):
    # search/download file and return path
    file = str(file)
    # file exists
    if os.path.isfile(file):
        return file
    # download
    elif file.startswith('http'):
        raise NotImplementedError()
        return file

    raise FileNotFoundError()


def load_yaml(path):
    # return Python dictionary
    file = check_file(path, mode='yaml')
    with open(file, 'r') as f:
        return yaml.safe_load(f)


def select_device(device):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    # return torch device
    pass