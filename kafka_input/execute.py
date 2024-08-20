import logging
import sys

from producer import EntityPairProducer
from consumer import EntityPairsConsumer
from utils.pool_logger import setup_logging

logger = logging.getLogger()
consumer_log_filename = "../logs/kafka-consumer-logs.log"
producer_log_filename = "../logs/kafka-producer-logs.log"


def run(args='producer'):
    if args == 'producer':
        setup_logging(producer_log_filename, True, logger)
        producer = EntityPairProducer()
        producer.send_entity_pairs()
    elif args == 'consumer':
        setup_logging(consumer_log_filename, True, logger)
        consumer = EntityPairsConsumer()
        consumer.receive_entity_pair_ids()
    else:
        logger.info("Provide proper argument. 'consumer' or 'producer'")


if __name__ == '__main__':
    # arg = 'consumer'
    arg = sys.argv[1]
    run(arg)
