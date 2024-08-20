import nlpaug.augmenter.word as naw
import math
import random

from config.config import get_aug_config


class WordAug:
    def __init__(self):
        aug_config = get_aug_config()
        self.aug_p = aug_config["aug_probs"]
        """self.del_probs = config_1["aug"]["del_probs"]
        self.aug_word_key_val = config_1["aug"]["aug_word_key_val"]
        self.aug_word_type = config_1["aug"]["aug_word_type"]
        self.del_ele = config_1["aug"]["del_ele"]
        self.aug_probs_key = config_1["aug"]["aug_probs_key"]"""

        self.synonym = naw.SynonymAug(
            aug_src='wordnet',
            name='Synonym_Aug',
            aug_p=self.aug_p,
            lang='eng',
            force_reload=False,
            verbose=0
        )
        self.spelling = naw.SpellingAug(
            name='Spelling_Aug',
            aug_p=self.aug_p,
            aug_min=1,
            aug_max=10,
            include_reverse=True,
            verbose=0
        )

    def remove_key(self, entity, del_probs):
        entity_aug = {}
        key_rem_perc = (del_probs * len(entity.keys()))
        key_rem_perc = math.ceil(key_rem_perc)
        entity_keys = list(set(entity.keys()))

        key_del = random.choices(entity_keys, k=key_rem_perc)

        for key, val in entity.items():
            if key not in key_del:
                entity_aug[key] = val

        return entity_aug

    def key_synonym_aug(self, entity, keys_aug):
        entity_aug = {}

        for key, val in entity.items():
            if key in keys_aug:
                key = str(key)
                aug_key = self.synonym.augment(key)
                if isinstance(aug_key, list):
                    aug_key = ''.join(aug_key)
                entity_aug[aug_key] = val
            else:
                entity_aug[key] = val

        return entity_aug

    def val_synonym_aug(self, entity, keys_aug):
        entity_aug = {}

        for key, val in entity.items():
            if key in keys_aug:
                val = str(val)
                aug_val = self.synonym.augment(val)
                if isinstance(aug_val, list):
                    aug_val = ''.join(aug_val)
                entity_aug[key] = aug_val
            else:
                entity_aug[key] = val

        return entity_aug

    def key_spelling_aug(self, entity, keys_aug):
        entity_aug = {}

        for key, val in entity.items():
            if key in keys_aug:
                key = str(key)
                aug_key = self.spelling.augment(key)
                if isinstance(aug_key, list):
                    aug_key = ''.join(aug_key)
                entity_aug[aug_key] = val
            else:
                entity_aug[key] = val

        return entity_aug

    def val_spelling_aug(self, entity, keys_aug):
        entity_aug = {}

        for key, val in entity.items():
            if key in keys_aug:
                val = str(val)
                aug_val = self.spelling.augment(val)
                if isinstance(aug_val, list):
                    aug_val = ''.join(aug_val)
                entity_aug[key] = aug_val
            else:
                entity_aug[key] = val

        return entity_aug

    def synonym_augmentation(self,
                             entity,
                             aug_probs_key,
                             aug_word_key_val):
        key_rem_perc = (aug_probs_key * len(entity.keys()))
        key_rem_perc = math.ceil(key_rem_perc)
        entity_keys = list(set(entity.keys()))

        keys_aug = random.choices(entity_keys, k=key_rem_perc)

        if aug_word_key_val == "KEY":
            entity_aug = self.key_synonym_aug(entity, keys_aug)
        elif aug_word_key_val == "VAL":
            entity_aug = self.val_synonym_aug(entity, keys_aug)
        elif aug_word_key_val == "BOTH":
            entity_aug = self.key_synonym_aug(entity, keys_aug)

            key_rem_perc = (aug_probs_key * len(entity_aug.keys()))
            key_rem_perc = math.ceil(key_rem_perc)
            entity_keys = list(set(entity_aug.keys()))

            keys_aug = random.choices(entity_keys, k=key_rem_perc)

            entity_aug = self.val_synonym_aug(entity, keys_aug)

        else:
            entity_aug = entity

        return entity_aug

    def spelling_augmentation(self,
                              entity,
                              aug_probs_key,
                              aug_word_key_val):

        key_rem_perc = (aug_probs_key * len(entity.keys()))
        key_rem_perc = math.ceil(key_rem_perc)
        entity_keys = list(set(entity.keys()))

        keys_aug = random.choices(entity_keys, k=key_rem_perc)

        if aug_word_key_val == "KEY":
            entity_aug = self.key_spelling_aug(entity, keys_aug)
        elif aug_word_key_val == "VAL":
            entity_aug = self.val_spelling_aug(entity, keys_aug)
        elif aug_word_key_val == "BOTH":
            entity_aug = self.key_spelling_aug(entity, keys_aug)

            key_rem_perc = (aug_probs_key * len(entity_aug.keys()))
            key_rem_perc = math.ceil(key_rem_perc)
            entity_keys = list(set(entity_aug.keys()))

            keys_aug = random.choices(entity_keys, k=key_rem_perc)

            entity_aug = self.val_spelling_aug(entity_aug, keys_aug)
        else:
            entity_aug = entity

        return entity_aug

    def word_augmentation(self,
                          entity,
                          aug_word_type,
                          aug_probs_key,
                          aug_word_key_val):
        if aug_word_type == "SYN":
            entity_aug = self.synonym_augmentation(entity,
                                                   aug_probs_key,
                                                   aug_word_key_val)
        elif aug_word_type == "SPL":
            entity_aug = self.spelling_augmentation(entity,
                                                    aug_probs_key,
                                                    aug_word_key_val)
        elif aug_word_type == "BOTH":
            entity_aug = self.synonym_augmentation(entity,
                                                   aug_probs_key,
                                                   aug_word_key_val)
            entity_aug = self.spelling_augmentation(entity_aug,
                                                    aug_probs_key,
                                                    aug_word_key_val)
        else:
            entity_aug = entity

        return entity_aug

    def execute_word_aug(self,
                         entity,
                         del_ele,
                         del_probs,
                         aug_word_type,
                         aug_probs_key,
                         aug_word_key_val):
        entities = [entity['entity_a'], entity['entity_b']]

        aug_entities = []
        for ent in entities:
            if del_ele:
                ent = self.remove_key(ent, del_probs)
            ent_af_aug = self.word_augmentation(ent,
                                                aug_word_type,
                                                aug_probs_key,
                                                aug_word_key_val)
            aug_entities.append(ent_af_aug)
        
        entity['entity_a'] = aug_entities[0]
        entity['entity_b'] = aug_entities[1]

        return entity


if __name__ == "__main__":
    wa = WordAug()
    entities = [
        {
            "_id": "10|20",
            "label_m": 0,
            "confidence": 0.97,
            "weight": 2.0,
            "entity_a": {
                "profile_id": "Match1436",
                "starring": "Kem Sereyvuth",
                "title": "City of Ghosts",
                "writer": "Mike Jones (screenwriter)"
            },
            "entity_b": {
                "profile_id": "Match1436",
                "year": "2002",
                "actor name": "Cheata, Ang",
                "director_name": "Dillon, Matt (I)",
                "genre": "Crime",
                "imdb_ksearch_id": "468495",
                "title": "City of Ghosts (2002)"
            }
        },
        {
            "_id": "10|21",
            "label_m": 0,
            "confidence": 0.89,
            "weight": 2.0,
            "entity_a": {
                "profile_id": "Match1436",
                "starring": "Kem Sereyvuth",
                "title": "City of Ghosts",
                "writer": "Mike Jones (screenwriter)"
            },
            "entity_b": {
                "profile_id": "Match1436",
                "year": "2002",
                "actor name": "Cheata, Ang",
                "director_name": "Dillon, Matt (I)",
                "genre": "Crime",
                "imdb_ksearch_id": "468495",
                "title": "City of Ghosts (2002)"
            },
        }
    ]
    entity = entities[0]
    print(entity)
    print(wa.execute_word_aug(entity))

