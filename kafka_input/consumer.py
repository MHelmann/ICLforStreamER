import sys
import requests
from kafka import KafkaConsumer
import json
import time

sys.path.append('../')

from pools.candidate_pool import *
from utils.kafka_util import entity_similarity_score
from config.config import get_kafka_config
from config.config import get_mongo_config
from config.config import get_deep_learning_config, get_aug_config
from meta_data.metadata import MetaData
from pools.entity_pool import EntityPool
from utils.model_deploy_util import ModelDeployFlag

logger = logging.getLogger(__name__)


def serialise_entity_pairs(entity_a, entity_b, weight, del_col, cols_rem_name):
    try:
        entities = [entity_a, entity_b]
        entity_pairs = []
        for entity in entities:
            entity_series = ""
            for key, val in entity.items():
                if del_col:
                    if key != "_id" and key not in cols_rem_name:
                        key = key.replace("\n", "")
                        val = val.replace("\n", "")

                        key = key.replace("\t", " ")
                        val = val.replace("\t", " ")

                        entity_series += f"COL {key} VAL {val} "
                else:
                    if key != "_id":
                        key = key.replace("\n", "")
                        val = val.replace("\n", "")

                        key = key.replace("\t", " ")
                        val = val.replace("\t", " ")

                        entity_series += f"COL {key} VAL {val} "
            entity_series = entity_series.strip()
            entity_pairs.append(entity_series)
        entity_serialized = '\t'.join(entity_pairs)

        return entity_serialized
    except Exception as args:
        raise f"Error while serialization of entity pairs. ERROR: {args}"


class EntityPairsConsumer:

    def __init__(self):
        kafka_config = get_kafka_config()
        mongo_config = get_mongo_config()
        self.model_flag = ModelDeployFlag()
        self.entity_pool = EntityPool()
        self.dl_config = get_deep_learning_config()
        self.pre_trained_model = self.dl_config["PRE_TRAINED_MODEL"]
        aug_config = get_aug_config()
        topic = kafka_config['topic']
        host = kafka_config['localhost']
        port = kafka_config['bootstrap_port']
        logger.info("\n")
        logger.info("*" * 40)
        try:
            self.consumer = KafkaConsumer(
                topic,
                bootstrap_servers=host + ':' + port,
                auto_offset_reset=kafka_config['auto_offset_reset'],
                group_id=kafka_config['group_a']
            )
        except Exception:
            logger.exception("Exception while connecting kafka-server.", exc_info=True)
            logger.info(f"Kafka details, HOST: {host}, PORT:{port}, TOPIC: {topic}")
            raise
        else:
            logger.info(f"Connected to kafka topic {topic}.")
            client = MongoClient(
                host=mongo_config['host'],
                port=mongo_config['port'],
                username="Helmann",
                password="helmanmm"
            )
            primary_db = client[mongo_config['primary_db']]
            self.entity_table_a = primary_db[mongo_config['entity_a']]
            self.entity_table_b = primary_db[mongo_config['entity_b']]
            self.ground_truth_table = primary_db[mongo_config['grd_truth_collection']]
            logger.info(f"Connected to {mongo_config['primary_db']} database.")

            candidate_pool_db = client[mongo_config['candidate_pool']]
            self.candidate_pool = candidate_pool_db[mongo_config['cand_collection']]
            logger.info(f"Connected to {mongo_config['candidate_pool']} database.")

            self.m_data = MetaData()
            self.meta_data = self.m_data.get_meta_data()
            self.num_rec_proc = mongo_config["num_rec_proc"]
            self.store_batch_size = mongo_config["store_batch_size"]

            # Remove Columns
            self.del_col = aug_config["del_col"]
            self.rem_col_names = aug_config["col_names"]
            logger.info(f"Initialized consumer")

    def store_data_candidate_pool(self, entity_pair_list):
        try:
            self.candidate_pool.insert_many(entity_pair_list)
            logger.info("Stored list of entities.")
        except Exception as arg:
            logger.exception("Exception while inserting entity_pool into candidate pool: primary collection.", exc_info=True)
            logger.error(arg)
            raise

    def receive_entity_pair_ids(self):
        """
        This function is kafka consumer.
        The msg containing entity_a_id, entity_b_id is read.
        The corresponding the entity details are retrieved from the database.
        The retrieved entity_pool is stored in the candidate pool along with corresponding label and confidence.
        :return: None
        """
        start_time = time.time()
        # store_start_time = time.time()
        # model_flag = self.model_flag.get_flag()
        flag = True
        entity_pair_list = []

        timestamp = datetime.utcnow()
        timestamp = datetime.timestamp(timestamp)
        self.meta_data["pool"]["last_store_ts"] = timestamp
        self.meta_data["label_pool"]["last_fetch_ts"] = timestamp
        self.meta_data["icl_pool"]["last_fetch_ts"] = timestamp
        self.m_data.set_meta_data(self.meta_data)
                    
        for idx, msg in enumerate(self.consumer):
            model_flag = self.model_flag.get_flag()
            if flag:
                logger.info("Started consumer entity pairs.")
                flag = False

            message = json.loads(msg.value)

            # The entity ids are retrieved from the message
            _id_a = message['entity_a']
            _id_b = message['entity_b']
            weight = message['pair_weight']

            # Entity detail is retrieved from the IMDB Profiles storage (entity_a)
            entity_a = self.entity_pool.get_entity_a(_id_a)

            # Entity detail is retrieved from the DBPedia profiles storage (entity_b)
            entity_b = self.entity_pool.get_entity_b(_id_b)

            serial_entity_pairs = serialise_entity_pairs(entity_a, entity_b, weight, self.del_col, self.rem_col_names)
            entity_pairs = {
                "entity_pairs": serial_entity_pairs
            }

            if model_flag == 1:
                try:
                    start_time_1 = time.time()
                    resp = requests.post("http://127.0.0.1:5001/predict", json=entity_pairs)
                    end_time_1 = time.time()
                except Exception as arg:
                    logger.exception("Inside FLASK-APP1.")
                    logger.error(arg)
                    raise arg
                else:
                    resp = resp.json()
                    label = resp["class"]
                    confidence = resp["confidence"]
            elif model_flag == 2:
                try:
                    start_time_1 = time.time()
                    resp = requests.post("http://127.0.0.1:5002/predict", json=entity_pairs)
                    end_time_1 = time.time()
                except Exception as arg:
                    logger.exception("Inside FLASK-APP1.")
                    logger.error(arg)
                    raise arg
                else:
                    resp = resp.json()
                    label = resp["class"]
                    confidence = resp["confidence"]
            else:
                try:
                    start_time_1 = time.time()
                    sim_score = entity_similarity_score(entity_a, entity_b, weight)
                    end_time_1 = time.time()
                except Exception as arg:
                    logger.exception("Inside Similarity Measure.")
                    logger.error(arg)
                    raise arg
                else:
                    label = None
                    confidence = sim_score
                    time.sleep(0.013)

            _id = str(_id_a) + "|" + str(_id_b) + "|" + str(idx + 1)

            timestamp = datetime.utcnow()
            timestamp = datetime.timestamp(timestamp)
            data = {
                "_id": _id,
                "label_m": label,
                "confidence": confidence,
                "weight": weight,
                "_id_a": _id_a,
                "_id_b": _id_b,
                "timestamp": timestamp
            }

            if (idx+1) % self.store_batch_size == 0:
                entity_pair_list.append(data)
                # store_end_time = time.time()
                self.store_data_candidate_pool(entity_pair_list)
                # logger.info(f"Time taken to store {idx+1} records: {store_end_time - store_start_time}")
                # store_start_time = time.time()
                entity_pair_list = []
            else:
                entity_pair_list.append(data)

            if (idx+1) % self.num_rec_proc == 0:
                end_time = time.time()
                logger.info(f"Time taken to consume {idx+1} records: {end_time - start_time}")
                logger.info(f"Consumed and Stored {idx+1} entity pairs to Candidate Pool.")
                logger.info(f"FLASK1-{end_time_1 - start_time_1}")
                logger.info(f"MODEL_FLAG: {model_flag}")
                start_time = time.time()

        logger.info("Closing the kafka-consumer.")
        self.close_consumer()

    def close_consumer(self):
        self.consumer.close()
