import logging

from pymongo import MongoClient
from config.config import get_mongo_config

logger = logging.getLogger(__name__)


class ModelDeployFlag:
    def __init__(self):
        mongo_config = get_mongo_config()
        client = MongoClient(
            host=mongo_config["host"],
            port=mongo_config["port"]
        )
        db = client["flags"]
        self.collection = db["flag_col"]
        logger.info("Connected to database config database.")

    def insert_flag(self, flag):
        query = {
            "_id": 1,
            "flag": flag
        }
        try:
            self.collection.insert_one(query)
        except Exception as arg:
            logger.exception("Exception occurred while inserting the flag.")
            logger.error(arg)
            raise

    def update_flag(self, flag):
        try:
            self.collection.find_one_and_update(
                {"_id": 1},
                {"$set": {"flag": flag}}
            )
        except Exception as arg:
            logger.exception("Exception occurred while updating the flag.")
            logger.error(arg)
            raise

    def get_flag(self):
        try:
            data = self.collection.find_one({"_id": 1})
            flag = data["flag"]
        except Exception as arg:
            logger.exception("Exception ocurred while returning flag.")
            logger.error(arg)
            raise
        else:
            return flag
