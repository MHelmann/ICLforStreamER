import json
from config.config import get_config
import logging

logger = logging.getLogger(__name__)


class EntityPool:
    def __init__(self):
        self.config = get_config()
        try:
            pool_a_file = open(self.config["pool_a_file"], "r", encoding="utf8")
            pool_b_file = open(self.config["pool_b_file"], "r", encoding="utf8")
            logger.info("Reading the entity_a, entity_b json files.")
        except Exception as args:
            logger.exception("Exception occurred while reading the json files.")
            logger.error(args)
            raise
        else:
            self.entity_a_data = json.load(pool_a_file)
            self.entity_b_data = json.load(pool_b_file)

            pool_a_file.close()
            pool_b_file.close()

            self.pool_a = self.get_pool_a()
            self.pool_b = self.get_pool_b()

    def get_pool_a(self):
        pool_a = dict()
        try:
            for item in self.entity_a_data:
                pool_a[item["_id"]] = item
            logger.info("Converted entity A entity_pool into dictionary.")
        except Exception as arg:
            logger.exception("Exception occurred while converting entity A json to dictionary.")
            logger.error(arg)
            raise
        else:
            return pool_a

    def get_pool_b(self):
        pool_b = dict()
        try:
            for item in self.entity_b_data:
                pool_b[item["_id"]] = item
            logger.info("Converted entity B entity_pool into dictionary.")
        except Exception as arg:
            logger.exception("Exception occurred while converting entity B json to dictionary.")
            logger.error(arg)
            raise
        else:
            return pool_b

    def get_entity_a(self, key):
        try:
            entity = self.pool_a[key]
        except Exception as args:
            logger.exception("Exception occurred while getting the entity.")
            logger.error(args)
            raise
        else:
            return entity

    def get_entity_b(self, key):
        try:
            entity = self.pool_b[key]
        except Exception as args:
            logger.exception("Exception occurred while getting the entity.")
            logger.error(args)
            raise
        else:
            return entity


if __name__ == "__main__":
    ep = EntityPool()
    print(ep.get_entity_a(12))
    print(ep.get_entity_b(130))
