import json


def load_configuration():
    fid = open('./config.json')
    return json.load(fid)


CONFIG = load_configuration()
