import json
import sys

sys.path.append('../')

config_file = "../config/config.json"

with open(config_file) as cf:
    config = json.load(cf)


def get_mongo_config():
    return config["mongodb"]


def store_mongo_config(mongo_config):
    config["mongodb"] = mongo_config
    conf = json.dumps(config, indent=4)
    json_file = open(config_file, "w")
    json_file.write(conf)
    json_file.close()


def get_config_file():
    return config_file


def get_config():
    return config


def get_classifier_config():
    return config["classifier"]


def get_kafka_config():
    return config["kafka"]


def get_deep_learning_config():
    return config["deep_learning"]


def get_aug_config():
    return config["aug"]
