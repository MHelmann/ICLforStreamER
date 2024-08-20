import datetime
import logging
import time

from pymongo import MongoClient
from augmentor.entity_char_aug import CharAugmentor
from augmentor.entity_word_aug import WordAug
from config.config import get_aug_config, get_deep_learning_config, get_mongo_config
from meta_data.metadata import MetaData
from utils.kafka_util import get_entity_details

logger = logging.getLogger(__name__)


def get_match_perc(match_count, total_count):
    try:
        match_perc = (match_count / total_count) * 100
    except Exception:
        logger.exception("Exception while calculating the matching entity pairs percentage.", exc_info=True)
        raise
    else:
        return match_perc


def get_num_unmatch(match_perc, total_count, match_count):
    unmatch_num = int((match_perc / 100) * total_count - match_count)
    if unmatch_num == 0:
        unmatch_num = 1
    return unmatch_num


class LabelPool:
    def __init__(self):
        mongo_config = get_mongo_config()
        dl_config = get_deep_learning_config()
        aug_config = get_aug_config()

        self.num_rec_retrieve = mongo_config["num_rec_retrieve"]

        self.duplicate_entity = aug_config["unmatch_entities_aug"]["duplicate_entity"]

        self.meta_data = MetaData()
        self.m_data = self.meta_data.get_meta_data()

        host = mongo_config["host"]
        port = mongo_config["port"]
        try:
            client = MongoClient(
                host=host,
                port=port,
                username="Helmann",
                password="helmanmm"
            )
        except Exception:
            logger.exception("Exception while connecting to label pool.", exc_info=True)
            logger.info(f"Connection details: host: {host}, port: {port}")
            raise
        else:
            primary_db_name = mongo_config["primary_db"]
            pool_a_name = mongo_config["entity_a"]
            pool_b_name = mongo_config["entity_b"]

            primary_db = client[primary_db_name]
            self.pool_a = primary_db[pool_a_name]
            self.pool_b = primary_db[pool_b_name]

            label_db_name = mongo_config["label_pool"]
            primary_coln_name = mongo_config["label_pool_col"]
            enriched_coln_name = mongo_config["label_secondary_col"]

            label_pool_db = client[label_db_name]
            self.primary_coln = label_pool_db[primary_coln_name]
            self.enriched_coln = label_pool_db[enriched_coln_name]

            logger.info(f"Connected to {label_db_name} database.")

            self.num_rec_retrieved = self.m_data["label_pool"]["num_record_retrieve"]
            self.th_match_perc = dl_config["th_match_perc"]
            self.word_augmentor = WordAug()
            self.char_augmentor = CharAugmentor()

            # Delete Columns
            self.del_col = aug_config["del_col"]
            self.col_names = aug_config["col_names"]

            # Augmentation configs for match(m) entities
            self.m_del_ele = aug_config["match_entities_aug"]["del_ele"]
            self.m_del_probs = aug_config["match_entities_aug"]["del_probs"]
            self.m_aug_probs_key = aug_config["match_entities_aug"]["aug_probs_key"]
            self.m_aug_word_key_val = aug_config["match_entities_aug"]["aug_word_key_val"]
            self.m_aug_word_type = aug_config["match_entities_aug"]["aug_word_type"]
            self.m_aug_char_key_val = aug_config["match_entities_aug"]["aug_char_key_val"]
            self.m_aug_char_type = aug_config["match_entities_aug"]["aug_char_type"]

            # Augmentation configs for unmatch(um) entities
            self.um_del_ele = aug_config["unmatch_entities_aug"]["del_ele"]
            self.um_del_probs = aug_config["unmatch_entities_aug"]["del_probs"]
            self.um_aug_probs_key = aug_config["unmatch_entities_aug"]["aug_probs_key"]
            self.um_aug_word_key_val = aug_config["unmatch_entities_aug"]["aug_word_key_val"]
            self.um_aug_word_type = aug_config["unmatch_entities_aug"]["aug_word_type"]
            self.um_aug_char_key_val = aug_config["unmatch_entities_aug"]["aug_char_key_val"]
            self.um_aug_char_type = aug_config["unmatch_entities_aug"]["aug_char_type"]

    def store(self, entity_pair_list):
        """
        This function stores the entities in label-pool's primary collection.
        :param entity: Entity consisting of entity-a, entity-b, and true label.
        """
        try:
            self.primary_coln.insert_many(entity_pair_list)
        except Exception:
            entity = entity_pair_list[-1]
            _id_a = entity["_id_a"]
            _id_b = entity["_id_b"]
            logger.exception(f"Exception while storing entity in Label Pool. "
                             f"Entity pair: {_id_a}, {_id_b}", exc_info=True)
            raise

    def transfer_entity_primary_seco(self):
        last_fetched_ts = self.m_data["label_pool"]["last_fetch_ts"]
        query = {
            "timestamp": {
                "$gte": last_fetched_ts
            }
        }

        logger.info(f"Query to tranfer entity pairs from label pool primary collection to secondary collection: "
                    f"{str(query)}")
        
        start_process_time = time.time()
        for doc in self.primary_coln.find(query).limit(self.num_rec_retrieve):
            last_fetched_ts = doc["timestamp"]
            doc["entity_a"] = get_entity_details(self.pool_a, doc["_id_a"])
            doc["entity_b"] = get_entity_details(self.pool_b, doc["_id_b"])
            self.enriched_coln.insert_one(doc)
        end_process_time = time.time()
        #self.m_data["label_pool"]["last_fetch_ts"] = last_fetched_ts 
        #self.meta_data.set_meta_data(self.m_data)

        logger.info("Transferred entities from Label Pool primary to secondary collection.")
        logger.info(f"Total number of entity pairs transferred: {self.num_rec_retrieve}")
        logger.info(f"Time taken to process {self.num_rec_retrieve} entity_pool-points: "
                    f"{end_process_time - start_process_time}")
        logger.info(f"Updated the last_fetched_timestamp to {datetime.datetime.fromtimestamp(last_fetched_ts)}")

    def get_total_count(self):
        return self.enriched_coln.count_documents({})

    def get_match_count(self):
        return self.enriched_coln.count_documents({"label_o": 1})

    def get_unmatch_count(self):
        return self.enriched_coln.count_documents({"label_o": 0})

    def get_match_entities(self):
        return self.enriched_coln.find({"label_o": 1})

    def get_unmatch_entities(self):
        return self.enriched_coln.find({"label_o": 0})

    def get_all_entities(self):
        return self.enriched_coln.find({})

    def truncate(self):
        self.enriched_coln.delete_many({})

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

    def get_aug_data(self):
        match_count = self.get_match_count()
        total_count = self.get_total_count()
        unmatch_count = self.get_unmatch_count()

        logger.info(f"Count of match-entity pairs: {match_count}. Unmatch-entity pair: {unmatch_count}")

        aug_unmatch_count = get_num_unmatch(self.th_match_perc, total_count, match_count)
        match_perc = get_match_perc(match_count, total_count)

        if 0 <= match_perc <= self.th_match_perc:
            augmented_unmatch_entities = []
            augmented_match_entities = []

            match_entities = list(self.get_match_entities())
            unmatch_entities = list(self.get_unmatch_entities())
            if self.del_col:
                try:
                    match_entities = self.remove_cols(match_entities)
                    unmatch_entities = self.remove_cols(unmatch_entities)
                except Exception as arg:
                    logger.exception("Exception raised while deleting the columns.")
                    logger.info(arg)
                    raise
                else:
                    logger.info("Removed columns mentioned in column names from entities.")
            unmatch_entities_aug = unmatch_entities[
                                   :aug_unmatch_count]  # Stores the unmatched entities for augmentation
            unmatch_entities_nanaug = unmatch_entities[
                                      aug_unmatch_count:]  # Stored the non-augmented unmatched entities

            logger.info(f"Count of match entity pairs for augmentation: {len(match_entities)}.")
            logger.info(f"Count of match entity pairs for augmentation: {len(unmatch_entities_aug)}.")
            logger.info("Augmenting the unmatched entity pairs...")
            for entity in unmatch_entities_aug:
                if self.duplicate_entity == "A":
                    entity["entity_b"] = entity["entity_a"]
                    entity["label_o"] = 1
                elif self.duplicate_entity == "B":
                    entity["entity_a"] = entity["entity_b"]
                    entity["label_o"] = 1
                else:
                    entity["entity_a"] = entity["entity_a"]
                    entity["entity_b"] = entity["entity_b"]
                    entity["label_o"] = entity["label_o"]
                
                aug_entity = self.word_augmentor.execute_word_aug(entity,
                                                                  self.um_del_ele,
                                                                  self.um_del_probs,
                                                                  self.um_aug_word_type,
                                                                  self.um_aug_probs_key,
                                                                  self.um_aug_word_key_val)
                aug_entity = self.char_augmentor.execute_char_aug(aug_entity,
                                                                  self.um_del_ele,
                                                                  self.um_del_probs,
                                                                  self.um_aug_char_type,
                                                                  self.um_aug_probs_key,
                                                                  self.um_aug_char_key_val)
                augmented_unmatch_entities.append(aug_entity)            
            logger.info("Augmenting matched entity pairs...")
            for entity in match_entities:
                aug_entity = self.word_augmentor.execute_word_aug(entity,
                                                                  self.m_del_ele,
                                                                  self.m_del_probs,
                                                                  self.m_aug_word_type,
                                                                  self.m_aug_probs_key,
                                                                  self.m_aug_word_key_val)
                aug_entity = self.char_augmentor.execute_char_aug(aug_entity,
                                                                  self.m_del_ele,
                                                                  self.m_del_probs,
                                                                  self.m_aug_char_type,
                                                                  self.m_aug_probs_key,
                                                                  self.m_aug_char_key_val)

                augmented_match_entities.append(aug_entity)
            logger.info("Augmentation of match entities completed...")
            
            entities = match_entities + unmatch_entities_nanaug + augmented_match_entities + augmented_unmatch_entities           
            match_count_aft_aug = len(match_entities) + len(augmented_match_entities) + len(augmented_unmatch_entities)
            unmatch_count_aft_aug = len(unmatch_entities_nanaug)
            logger.info(f"After augmentation, Match-count: {match_count_aft_aug}, "
                        f"and Unmatch-count: {unmatch_count_aft_aug}")
        else:
            logger.info(f"The match class percentage is more than threshold: {self.th_match_perc}."
                        f" Augmentation not required.")
            entities = list(self.get_all_entities())
            if self.del_col:
                entities = self.remove_cols(entities)
                logger.info("Removed columns mentioned in column names from entities.")
            num_records = len(entities)
            logger.info(f"Total number of entity_pool-points selected for training: {num_records}")
        return entities

    def get_entities(self):

        um_entity_pairs = list(self.get_unmatch_entities())
        m_entity_pairs = list(self.get_match_entities())

        entities = um_entity_pairs + m_entity_pairs

        return entities

    def retrieve_data(self, da):
        """
        This function transfers the entity pairs from label-pool's primary collection to secondary collection.
        It also augments the entity_pool if augmentation is set to True.
        :param da: Boolean flag for augmentation.
        :return : The augmented entity_pool if da: True else un-augmented entity_pool.
        """
        logger.info("Retrieving entity_pool from a label pool for the purpose of testing and training deep learning models.")
        logger.info("Truncating label pool secondary collection.")

        self.truncate()
        self.transfer_entity_primary_seco()

        if da:
            start_time = time.time()
            data = self.get_aug_data()
            end_time = time.time()
            logger.info(f"Time taken for AUGMENTATION: {end_time - start_time}")
        else:
            match_count = self.get_match_count()
            unmatch_count = self.get_unmatch_count()
            logger.info(f"Data Augmentation Flag is FALSE. So not applying augmentor.")
            logger.info(f"Count of match entity-pairs: {match_count}, non-match entity-pair: {unmatch_count}")
            data = list(self.get_all_entities())
            if self.del_col:
                data = self.remove_cols(data)
                logger.info("Removed columns mentioned in column names from entities.")
        return data


if __name__ == "__main__":
    lp = LabelPool()
    lp.retrieve_data(True)
