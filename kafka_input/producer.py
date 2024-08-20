import json
import logging
import sys
from kafka import KafkaProducer

sys.path.append('../')
from config.config import get_kafka_config
from config.config import get_mongo_config

logger = logging.getLogger(__name__)


def json_serializer(data):
    return json.dumps(data).encode("utf-8")


class EntityPairProducer:
    def __init__(self):
        self.kafka_config = get_kafka_config()
        host = self.kafka_config['localhost']
        port = self.kafka_config['bootstrap_port']
        topic = self.kafka_config['topic']
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=host + ':' + port,
                value_serializer=json_serializer)
        except Exception as arg:
            logger.exception("Exception while connecting to kafka-producer.")
            logger.info(f"Kafka-producer details: HOST: {host}, PORT: {port}, TOPIC: {topic}")
            logger.error(arg)
        else:
            self.mongo_config = get_mongo_config()
            logger.info(f"Connected to kafka topic: {topic}.")
            try:
                with open(self.kafka_config['input']) as file:
                    self.lines = file.readlines()
                file.close()
                logger.info(self.lines[0])
            except Exception:
                logger.exception("Exception while opening the input file.")

    @staticmethod
    def get_entity_pair_ids(entity_a_idx, entity_b_idx, pair_weight):
        return {
            'entity_a': entity_a_idx,
            'entity_b': entity_b_idx,
            'pair_weight': pair_weight
        }

    def send_entity_pairs(self):
        logger.info("Started producing the entity-pairs.")

        for i, line in enumerate(self.lines):
            line = line.strip()
            line = line.split(",")
            entity_pair_ids = self.get_entity_pair_ids(int(line[0]), int(line[1]), float(line[2]))
            self.producer.send(self.kafka_config['topic'], entity_pair_ids)
            if i % self.mongo_config["num_rec_proc"] == 0:
                logger.info(f"Produced {i} entity-pairs.")
        logging.info("Produced all entity-pairs.")
        self.close_producer()

    def close_producer(self):
        logger.info("Closing the kafka-producer.")
        self.producer.flush()
        self.producer.close()
