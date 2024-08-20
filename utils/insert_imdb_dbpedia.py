import json
import sys
sys.path.append("../")

from pymongo import MongoClient
from config.config import get_mongo_config, get_config


class InsertDatasetToDB:
    """
    This class creates the database in MongoDB and stores the data from .json file to databse collections.
    """
    def __init__(self):
        self.mongo_config = get_mongo_config()
        self.config = get_config()

        client = MongoClient(
            host=self.mongo_config["host"],
            port=self.mongo_config["port"]
        )

        primary = client[self.mongo_config["primary_db"]]
        self.grd_truth_col = primary[self.mongo_config["grd_truth_collection"]]
        self.entity_a_pool = primary[self.mongo_config["entity_a"]]
        self.entity_b_pool = primary[self.mongo_config["entity_b"]]

        self.pool_a_file = open(self.config["pool_a_file"])
        self.pool_b_file = open(self.config["pool_b_file"])
        self.grd_truth_file = open(self.config["ground_truth_file"])

    def change_idx(self, json_data):
        for i, data in enumerate(json_data):
            if type(data['idx']) is str:
                data['idx'] = i
            data["_id"] = data["idx"]
            del data["idx"]
        return json_data

    def insert_data(self):
        pool_a_data = json.load(self.pool_a_file)
        pool_b_data = json.load(self.pool_b_file)
        grd_data = json.load(self.grd_truth_file)
        print("Loaded the data from json file.")

        # pool_a_data = self.change_idx(pool_a_data)    # Use this function if the id key is "idx" instead of "_id" in JSON file
        # pool_b_data = self.change_idx(pool_b_data)    # Use this function if the id key is "idx" instead of "_id" in JSON file
        # print("Changed the idx to _id")

        self.entity_a_pool.insert_many(pool_a_data)
        self.entity_b_pool.insert_many(pool_b_data)
        self.grd_truth_col.insert_many(grd_data)
        print("Inserted the data in MongoDB")
        print("SUCCESSFULL!")


if __name__ == "__main__":
    insert_data = InsertDatasetToDB()
    insert_data.insert_data()
