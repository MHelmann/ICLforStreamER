import logging
from datetime import datetime
from pymongo import MongoClient

from config.config import get_mongo_config

logger = logging.getLogger(__name__)


def store(_id_a,
          _id_b,
          candidate_table,
          matcher_label=None,
          matcher_confidence=None,
          weight=None):

    _id = str(_id_a) + "|" + str(_id_b) + "|" + str(weight)

    timestamp = datetime.utcnow()
    timestamp = datetime.timestamp(timestamp)
    data = {
        "_id": _id,
        "label_m": matcher_label,
        "confidence": matcher_confidence,
        "weight": weight,
        "_id_a": _id_a,
        "_id_b": _id_b,
        "timestamp": timestamp
    }
    try:
        candidate_table.insert_one(data)
    except Exception as arg:
        logger.exception("Exception while inserting entity_pool into candidate pool: primary collection.", exc_info=True)
        logger.error(arg)
        raise


def drop_collection(db, collection):
    db.drop_collection(collection)


if __name__ == "__main__":
    mongo_config = get_mongo_config()
    client = MongoClient(
        host=mongo_config['host'],
        port=mongo_config['port']
    )
    candidate_pool_db = client[mongo_config['candidate_pool']]
    candidate_pool = candidate_pool_db[mongo_config['cand_collection']]

    entity_a_id = 1000
    entity_b_id = 2100

    entity_a = {
        "profile_id": "Match1436",
        "starring": "Kem Sereyvuth",
        "title": "City of Ghosts",
        "writer": "Mike Jones (screenwriter)"
    }

    entity_b = {
        "profile_id": "Match1436",
        "year": "2002",
        "actor_name": "Cheata, Ang",
        "director_name": "Dillon, Matt (I)",
        "genre": "Crime",
        "imdb_ksearch_id": "468495",
        "title": "City of Ghosts (2002)"
    }
    label = 0
    confidence = 0.99
    weight = 0.1

    # table = config_1['mongodb']['cand_collection']

    # drop_collection(table)

    store(
        entity_a_id,
        entity_b_id,
        candidate_pool,
        label,
        confidence,
        weight
    )
