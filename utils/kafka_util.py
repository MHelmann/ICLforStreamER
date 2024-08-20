from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.deep_learning_util import serialise_entity_pair
import logging

logger = logging.getLogger(__name__)


def data_for_classifier(entity_a, entity_b, weight):
    data = {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "weight": weight
    }

    return data


def entity_similarity_score(entity_a, entity_b, weight):
    # entities = []
    serial_entities = serialise_entity_pair(entity_a, entity_b, weight)
    vectorizer = TfidfVectorizer(analyzer='word', norm=None, use_idf=True, smooth_idf=True)

    tfIdfMat = vectorizer.fit_transform(serial_entities)
    csim = cosine_similarity(tfIdfMat, tfIdfMat)
    sim_score = csim[0][1]

    return sim_score


def get_entity_details(collection, entity_idx):
    """
    :param collection: table name (imdb or dbpedia)
    :param entity_idx:  index of the entity
    :return: entity detail
    """
    try:
        query = {
            "_id": entity_idx
        }
        entity = collection.find_one(query)
    except Exception:
        logger.exception("Exception occurred while retrieving entity pair.", exc_info=True)
        raise
    else:
        return entity


def get_label(ground_truth_col, _id_a, _id_b):
    """
    This function finds the ground truth label for given entity pairs.
    :param ground_truth_col: Ground-truth collection name.
    :param _id_a: Entity-a id.
    :param _id_b: Entity-b id.
    :return : Ground truth label of entity pairs.
    """
    _id_a = int(_id_a)
    _id_b = int(_id_b)
    query = {
        "$and": [
            {"entity_a": _id_a},
            {"entity_b": _id_b}
        ]
    }

    try:
        entity_label = ground_truth_col.find_one(query)
    except Exception:
        logger.exception(f"Exception while finding the ground truth label "
                         f"for entity pair {_id_a}, {_id_b}", exc_info=True)
        raise
    else:
        if entity_label is None:
            return 0
        else:
            return 1


if __name__ == "__main__":
    weight = 1.0

    entity_a = {
        "_id": 10,
        "profile_id": "Match1436",
        "starring": "Kem Sereyvuth",
        "title": "City of Ghosts",
        "writer": "Mike Jones (screenwriter)"
    }

    entity_b = {
        "_id": 20,
        "profile_id": "Match1436",
        "year": "2002",
        "actor_name": "Cheata, Ang",
        "director_name": "Dillon, Matt (I)",
        "genre": "Crime",
        "imdb_ksearch_id": "468495",
        "title": "City of Ghosts (2002)"
    }

    sim_score = entity_similarity_score(entity_a, entity_b, weight)
    print(sim_score)
