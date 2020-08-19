import json
import os


def load_configuration():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    fid = open(os.path.join(current_folder, './config.json'))
    return json.load(fid)


CONFIG = load_configuration()
