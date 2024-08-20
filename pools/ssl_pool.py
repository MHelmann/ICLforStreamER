import logging
import time
import math
from random import shuffle

from pymongo import MongoClient
from config.config import get_aug_config, get_mongo_config
from meta_data.metadata import MetaData
from utils.kafka_util import get_entity_details

logger = logging.getLogger(__name__)


class SSLPool:
    def __init__(self):
        mongo_config = get_mongo_config()
        aug_config = get_aug_config()

        self.meta_data = MetaData()
        self.m_data = self.meta_data.get_meta_data()

        host = mongo_config["host"]
        port = mongo_config["port"]
        try:
            client = MongoClient(
                host=host,
                port=port
            )
        except Exception:
            logger.exception("Exception while connecting to candidate pool.", exc_info=True)
            logger.info(f"Connection details: host: {host}, port: {port}")
            raise
        else:
            primary_db_name = mongo_config["primary_db"]
            pool_a_name = mongo_config["entity_a"]
            pool_b_name = mongo_config["entity_b"]
            primary_db = client[primary_db_name]
            self.pool_a = primary_db[pool_a_name]
            self.pool_b = primary_db[pool_b_name]

            cand_db_name = mongo_config["candidate_pool"]
            second_coln_name = mongo_config["secondary_pool"]
            cand_pool_db = client[cand_db_name]
            self.second_coln = cand_pool_db[second_coln_name]
            logger.info(f"Connected to {cand_db_name} database.")

            label_db_name = mongo_config["label_pool"]
            primary_coln_name = mongo_config["label_pool_col"]
            label_pool_db = client[label_db_name]
            self.primary_coln = label_pool_db[primary_coln_name]
            logger.info(f"Connected to {label_db_name} database.")

            # Delete Columns
            self.del_col = aug_config["del_col"]
            self.col_names = aug_config["col_names"]

    def get_all_entities(self, pool_flag):
        if pool_flag == 0:
            data = []
            last_fetched_ts = self.m_data["icl_pool"]["last_fetch_ts"]
            query = {
                "timestamp": {
                    "$gte": last_fetched_ts
                }
            }
            if self.second_coln.count_documents(query) >= 4000:
                
                # Sorts the secondary collection in descending order of confidence.
                # Then selects first 40% and last 40% and 20% in middle records.
                num_rec_to_label = 1000
                num_first_n_records = math.ceil(0.4 * num_rec_to_label)
                num_last_n_records = math.floor(0.4 * num_rec_to_label)
                num_middle_first_n_records = math.ceil(0.1 * num_rec_to_label)
                num_middle_last_n_records = math.ceil(0.1 * num_rec_to_label)

                total_records = list \
                        (
                        self.second_coln.find({}).sort([("confidence", -1)])
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

                entities_sorted = sorted(entity_pairs, key=lambda x: x['timestamp'])

                logger.info(f"Start retrieving entity pairs from candidate pool secondary collection for SSL training.")
                for doc in entities_sorted:
                    last_fetched_ts = doc["timestamp"]
                    doc["entity_a"] = get_entity_details(self.pool_a, doc["_id_a"])
                    doc["entity_b"] = get_entity_details(self.pool_b, doc["_id_b"])
                    data.append(doc)
                
                for doc_ in self.primary_coln.find(query):
                    last_fetched_ts = doc_["timestamp"]
                self.m_data["icl_pool"]["last_fetch_ts"] = last_fetched_ts
                self.meta_data.set_meta_data(self.m_data)
                 
                return data
            else:
                logger.info(f"Start retrieving entity pairs from candidate pool secondary collection for SSL training.")
                for doc in self.second_coln.find(query):
                    last_fetched_ts = doc["timestamp"]
                    doc["entity_a"] = get_entity_details(self.pool_a, doc["_id_a"])
                    doc["entity_b"] = get_entity_details(self.pool_b, doc["_id_b"])
                    data.append(doc)
                
                for doc_ in self.primary_coln.find(query):
                    last_fetched_ts = doc_["timestamp"]
                self.m_data["icl_pool"]["last_fetch_ts"] = last_fetched_ts
                self.meta_data.set_meta_data(self.m_data)
                
                return data
        elif pool_flag == 1:
            data = []
            data_old = []
            data_new = []
            last_fetched_ts = self.m_data["icl_pool"]["last_fetch_ts"]
            query_old = {
                "timestamp": {
                    "$lte": last_fetched_ts
                }
            }
            query_new = {
                "timestamp": {
                    "$gt": last_fetched_ts
                }
            }

            logger.info(f"Query to retrieve new and old entity pairs from label pool primary collection for ICL training.")

            start_process_time = time.time()
            for doc in self.primary_coln.find(query_old):
                doc["entity_a"] = get_entity_details(self.pool_a, doc["_id_a"])
                doc["entity_b"] = get_entity_details(self.pool_b, doc["_id_b"])
                data_old.append(doc)
            for doc in self.primary_coln.find(query_new):
                last_fetched_ts = doc["timestamp"]
                doc["entity_a"] = get_entity_details(self.pool_a, doc["_id_a"])
                doc["entity_b"] = get_entity_details(self.pool_b, doc["_id_b"])
                data_new.append(doc)          
            end_process_time = time.time()
            self.m_data["icl_pool"]["last_fetch_ts"] = last_fetched_ts
            self.meta_data.set_meta_data(self.m_data)
            logger.info(f"Time taken to move entities for SSL trainig: {end_process_time - start_process_time}")
            
            data.append(data_old)
            data.append(data_new)
            return data
        
    def remove_cols(self, entity_pair_list):
        updated_entity_list = []
        for entity in entity_pair_list:
            entity_a = entity["entity_a"]
            entity_b = entity["entity_b"]
            updated_entity_a = dict()
            updated_entity_b = dict()
            for key, val in entity_a.items():
                if key in self.col_names:
                    continue
                else:
                    updated_entity_a[key] = val

            for key, val in entity_b.items():
                if key in self.col_names:
                    continue
                else:
                    updated_entity_b[key] = val

            entity["entity_a"] = updated_entity_a
            entity["entity_b"] = updated_entity_b
            updated_entity_list.append(entity)

        return updated_entity_list

    def retrieve_data(self, pool_flag):
        """
        This function retrieves all entities from the candidate pool secondary collections during first iteration.
        For all following-up collections data samples are retrieved from labeled pool primary collection.
        Further, columns get deleted if del_col is set to True.
        :return : List of entities that are enreached with corresponding entity description; if del_col: True --> delete specified columns.
        """
        logger.info("Retrieving entities from label pool primary colleciton for the purpose of ssl/icl training.")

        start_time = time.time()
        data = self.get_all_entities(pool_flag)
        if self.del_col:
            if pool_flag == 0:
                data = self.remove_cols(data)
                logger.info("Removed columns mentioned in column names from entities.")
                num_records = len(data)
                logger.info(f"Total number of candidate pool secondary collection points selected for ssl training: {num_records}")
            elif pool_flag == 1 or pool_flag == 2:
                data[0] = self.remove_cols(data[0])
                data[1] = self.remove_cols(data[1])
                logger.info("Removed columns mentioned in column names from entities.")
                num_records = len(data[0]) + len(data[1])
                logger.info(f"{len(data[0])} old  + {len(data[1])} new label pool primary collection points selected for training.")
        end_time = time.time()
        logger.info(f"Time taken for retrieving: {end_time - start_time}")

        return data


if __name__ == "__main__":
    lp = SSLPool()
    lp.retrieve_data(1)

