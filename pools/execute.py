import datetime
import subprocess
import os
import sys
import time
import logging
import math
from random import shuffle
from csv import writer

sys.path.append('../')

from pymongo import MongoClient
from utils.kafka_util import get_label
from DLEM.initial_training import EMStreamTrain
from pools.label_pool import LabelPool
from config.config import get_config_file, get_mongo_config, get_deep_learning_config
from meta_data.metadata import MetaData
from utils.kafka_util import get_entity_details
from utils.pool_logger import setup_logging
from utils.model_deploy_util import ModelDeployFlag

logger = logging.getLogger()
log_filename = "../logs/pool_logs.log"


class ActiveLabel:
    def __init__(self):
        self.config_file = get_config_file()
        self.mongo_config = get_mongo_config()
        self.dl_config = get_deep_learning_config()
        self.model_flag = ModelDeployFlag()
        try:
            client = MongoClient(
                host=self.mongo_config["host"],
                port=self.mongo_config["port"]
            )
        except Exception:
            logger.exception("Exception occurred while connecting to Mongo Database.", exc_info=True)
            raise
        else:

            primary = self.mongo_config["primary_db"]
            grd_truth_col = self.mongo_config["grd_truth_collection"]
            entity_a_pool = self.mongo_config["entity_a"]
            entity_b_pool = self.mongo_config["entity_b"]

            primary_db = client[primary]
            self.ground_truth_col = primary_db[grd_truth_col]
            self.pool_a = primary_db[entity_a_pool]
            self.pool_b = primary_db[entity_b_pool]
            information = f"Connected to {primary} and {grd_truth_col} collection..."
            logger.info(information)
            
            candidate_pool = self.mongo_config["candidate_pool"]
            cand_col = self.mongo_config["cand_collection"]
            sec_pool = self.mongo_config["secondary_pool"]

            cp_db = client[candidate_pool]
            self.cand_pool = cp_db[cand_col]
            self.secondary_pool = cp_db[sec_pool]
            information = f"Connected to {candidate_pool} database and {cand_col}, {sec_pool} collection..."
            logger.info(information)

            self.num_rec_retrieve = self.mongo_config["num_rec_retrieve"]
            self.dl = EMStreamTrain()
            self.lp = LabelPool()
            self.m_data = MetaData()
            self.metadata = self.m_data.get_meta_data()
            self.strategy = self.mongo_config["data_selection_strategy"]

    def transfer_entity_pairs(self):
        """
        This function transfers the entity pairs between specified time-stamp to CP secondary collection.
        """
        last_stored_ts = self.metadata["pool"]["last_store_ts"]
        n_min = self.mongo_config["time_window_min"]

        last_stored_datetime = datetime.datetime.fromtimestamp(last_stored_ts)
        logger.info(f"Last fetched records datetime: {last_stored_datetime}")

        start_timestamp = last_stored_ts
        end_timestamp = last_stored_ts + (60 * n_min)

        query = {
            'timestamp': {
                '$gt': start_timestamp,
                '$lte': end_timestamp
            }
        }
        start_timestamp_date_time = datetime.datetime.fromtimestamp(start_timestamp)
        end_timestamp_date_time = datetime.datetime.fromtimestamp(end_timestamp)
        logger.info(f"Query: Fetches the records between {start_timestamp_date_time} and {end_timestamp_date_time}")
        try:
            start_process_time = time.time()
            for doc in self.cand_pool.find(query):
                self.secondary_pool.insert_one(doc)
            end_process_time = time.time()
        except Exception:
            logger.exception("Exception occurred in candidate pool while transferring the pairs from "
                             "primary collection to secondary collection", exc_info=True)
            raise
        else:
            info = f"Transferred the entity pairs between {datetime.datetime.fromtimestamp(start_timestamp)} " \
                   f"and {datetime.datetime.fromtimestamp(end_timestamp)} to secondary collection."
            logger.info(info)
            logger.info(f"Time taken to transfer time from candidate_pool primary to "
                        f"secondary pool: {end_process_time - start_process_time}")

            num_tranfer_entity_pairs = self.cand_pool.count_documents(query)
            info = f"Total number of entity pairs transferred: {num_tranfer_entity_pairs}"
            logger.info(info)

            self.metadata["pool"]["last_store_ts"] = end_timestamp
            self.m_data.set_meta_data(self.metadata)

            info = f"Updated the last stored timestamp to {datetime.datetime.fromtimestamp(end_timestamp)}"
            logger.info(info)

    def check_datapoints_cp(self):
        last_stored_ts = self.metadata["pool"]["last_store_ts"]
        n_min = self.mongo_config["time_window_min"]
    
        if self.metadata["time_range_flag"]:
            n_min = n_min / 10

        timestamp = last_stored_ts + (60 * n_min)

        try:
            datapoint = list(self.cand_pool.find().sort("timestamp", -1).limit(1))[0]
        except IndexError as arg:
            return False
        else:
            if datapoint["timestamp"] > timestamp:
                return True
            else:
                return False

    def get_data_to_label(self):
        """
        This function applies strategy to select sample from population.
        Strategy: The entity pairs are sorted based on weight and confidence in descending order.
                  Then selects first n entity_pool-points.
        :return : The sample of size 'num_rec_retrieve'.
        """
        try:
            if self.strategy == 1:
                # Sorts the entity_pool in descending order of confidence and weight.
                # Then select first "n" records.
                entity_pairs = list \
                        (
                        self.secondary_pool.find({}).sort([("confidence", -1), ("weight", -1)]).limit(
                            self.num_rec_retrieve)
                    )
                entity_pairs = [dict(t) for t in {tuple(d.items()) for d in entity_pairs}]
                logger.info(f"Applied strategy 1 to retrieve records for labelling.")
            elif self.strategy == 2:
                # Sorts the entity_pool in descending order of confidence.
                # Then selects first "n/2" first and "n/2" last records.
                upper_pairs = list \
                        (
                        self.secondary_pool.find({}).sort([("confidence", -1)]).limit(
                            math.ceil(self.num_rec_retrieve / 2))
                    )
                lower_pairs = list \
                        (
                        self.secondary_pool.find({}).sort([("confidence", 1)]).limit(
                            math.floor(self.num_rec_retrieve / 2))
                    )
                entity_pairs = upper_pairs + lower_pairs
                entity_pairs = [dict(t) for t in {tuple(d.items()) for d in entity_pairs}]
                logger.info(f"Applied strategy 2 to retrieve records for labelling.")
            elif self.strategy == 3:
                # Sorts the entity_pool in descending order of confidence.
                # Then selects first 40% and last 40% and 20% in middle records.
                num_rec_to_label = self.num_rec_retrieve
                num_first_n_records = math.ceil(0.4 * num_rec_to_label)
                num_last_n_records = math.floor(0.4 * num_rec_to_label)
                num_middle_first_n_records = math.ceil(0.1 * num_rec_to_label)
                num_middle_last_n_records = math.ceil(0.1 * num_rec_to_label)

                total_records = list \
                        (
                        self.secondary_pool.find({}).sort([("confidence", -1)])
                    )

                first_half_rec = total_records[:int(num_rec_to_label / 2)]
                second_half_rec = total_records[-int(num_rec_to_label / 2):]

                first_n_rec = total_records[:num_first_n_records]
                last_n_rec = total_records[-num_last_n_records:]

                middle_first_n_rec = first_half_rec[-num_middle_first_n_records:]
                middle_last_n_rec = second_half_rec[:num_middle_last_n_records]

                entity_pairs = first_n_rec + last_n_rec + middle_first_n_rec + middle_last_n_rec
                shuffle(entity_pairs)
                entity_pairs = [dict(t) for t in {tuple(d.items()) for d in entity_pairs}]
                logger.info(f"Applied strategy 3 to retrieve records for labelling.")
            else:
                # Sorts the entity_pool descending order of weight and confidecne.
                # Then selects first "n" records.
                entity_pairs = list \
                        (
                        self.secondary_pool.find({}).sort([("weight", -1), ("confidence", -1)]).limit(
                            self.num_rec_retrieve)
                    )
                entity_pairs = [dict(t) for t in {tuple(d.items()) for d in entity_pairs}]
                logger.info(f"Applied strategy 4 to retrieve records for labelling.")

        except Exception:
            logger.exception("Exception while fetching the pairs to label.", exc_info=True)
            raise
        else:
            logger.info(f"Selected first {self.num_rec_retrieve} entity pairs for labelling.")
            return entity_pairs

    def get_label_oracle(self, _id_a, _id_b):
        entity_a = get_entity_details(self.pool_a, _id_a)
        entity_b = get_entity_details(self.pool_b, _id_b)

        print(f"Entity A: {entity_a}")
        print(f"Entity B: {entity_b}")
        print(f"\n Are the above entity_a {_id_a} and entity_b {_id_b} match or not-match?")
        label = int(input("YES: 1    NO: 0    Your Choice: "))
        if label in [0, 1]:
            label = label
        else:
            label = 0

        return label

    def get_candidate_pool_count(self):
        last_stored_ts = self.metadata["pool"]["last_store_ts"]
        query = {
            "timestamp": {
                "$gte": last_stored_ts
            }
        }

        doc_counts = self.cand_pool.count_documents(query)
        return doc_counts

    def execute_labelling(self, dl_error_csv_obj, train_error_csv_obj, i):
        """
        This function transfers the entity_pool points from candidate-pool primary collection to secondary collection.
        It also assigns true labels for entity pairs and then store them in label-pool primary collection.
        :param dl_error_csv_obj: A csv-file to store deep-learning errors per epoch.
        :param train_error_csv_obj: A csv-file to store deep-learning training errors.
        :param i: Index.
        :return : None
        """
        
        # Delete secondary collection of candidate pool.
        self.secondary_pool.delete_many({})
        logger.info("Truncated the candidate secondary pool.")

        # Entity pairs are transferred to secondary collection.
        self.transfer_entity_pairs()

        # Applies the sampling strategy and retrieves the sample from secondary-collection.
        entity_pairs = self.get_data_to_label()
        logger.info(f"Length of entity_pairs list: {len(entity_pairs)}")
        logger.info("Starting labelling the entity pairs...")
        start_time = time.time()
        entity_pair_list = []
        for entity in entity_pairs:
            _id_a = entity["_id_a"]
            _id_b = entity["_id_b"]

            true_label = get_label(self.ground_truth_col, _id_a, _id_b)  # Assigns true-label.

            entity["label_o"] = true_label
            timestamp = datetime.datetime.utcnow()
            timestamp = datetime.datetime.timestamp(timestamp)
            entity["timestamp"] = timestamp

            entity_pair_list.append(entity)
            # self.lp.store(entity)  # Store the entity-pairs in label-pool's primary collection.

        self.lp.store(entity_pair_list)
        end_time = time.time()
        logger.info(f"Time taken to label the entity pairs: {end_time - start_time}")
        logger.info("Completed labelling the entity pairs and stored in the label pool.")
        
        # DEEP-LEARNING COMPONENT
        self.dl.execute_model_training(dl_error_csv_obj, train_error_csv_obj, i)


def execute():
    """
    This function creates instance of active label class.
    Waits till the required amount of datasets collected in candidate pool.
    """
    setup_logging(log_filename, True, logger)
    logger.info("\n")
    logger.info("-" * 40)
    oracle = ActiveLabel()
    if os.path.exists(oracle.dl_config["ewc_pickle_path"]):
        os.remove(oracle.dl_config["ewc_pickle_path"])
    i = 0
    n_min = oracle.mongo_config["time_window_min"]
    while True:
        logger.info("\n")
        logger.info("*" * 40)
        logger.info(f"Loop: {i}")
        logger.info("Starting active labelling...")
        count = 0
        dl_metadata_file = open("../output/dl_errors.csv", "a", newline='')
        train_error_file = open("../output/train_error.csv", "a", newline='')
        dl_error_csv_obj = writer(dl_metadata_file)
        train_error_csv_obj = writer(train_error_file)
        """if oracle.metadata["time_range_flag"]:
            logger.info("Dividing time range by ten.")
            n_min = n_min / 10"""
        while True:
            if oracle.check_datapoints_cp():
                break
            else:
                logger.info("Waiting for required amount of entity_pool-points to be collected in candidate pool.")
                time.sleep(int((n_min / 10) * 60))
                count += 1
                if count >= 5:
                    break
        if oracle.get_candidate_pool_count() > 10:
            oracle.execute_labelling(dl_error_csv_obj, train_error_csv_obj, i)
            i += 1
            logger.info("*" * 40)
        else:
            logger.info("No entity_pool points in the candidate pool")
            logger.info("*" * 40)
            break
        dl_metadata_file.close()
        train_error_file.close()

    exit_code = subprocess.Popen("./../classifier/shell_script/app1_stop.sh", shell=True,
                                 stdout=subprocess.PIPE)
    subprocess_return = exit_code.stdout.read()
    logger.info(subprocess_return)

    exit_code = subprocess.Popen("./../classifier/shell_script/app2_stop.sh", shell=True,
                                 stdout=subprocess.PIPE)
    subprocess_return = exit_code.stdout.read()
    logger.info(subprocess_return)


if __name__ == "__main__":
    execute()
