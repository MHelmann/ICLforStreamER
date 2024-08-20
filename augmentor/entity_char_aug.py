import nlpaug.augmenter.char as nac
import math
import random

from config.config import get_aug_config


class CharAugmentor:
    def __init__(self):
        aug_config = get_aug_config()
        self.aug_p = aug_config["aug_probs"]
        self.del_probs = aug_config["del_probs"]
        """self.aug_char_key_val = config_1["aug"]["aug_char_key_val"]
        self.aug_char_type = config_1["aug"]["aug_char_type"]
        self.del_ele = config_1["aug"]["del_ele"]
        self.aug_probs_key = config_1["aug"]["aug_probs_key"]"""

        self.keyboard_aug = nac.KeyboardAug(
            name='Keyboad_aug',
            aug_char_p=self.aug_p,
            aug_char_min=1,
            aug_char_max=10,
            include_special_char=False,
            include_numeric=False,
            include_upper_case=True,
            lang='en',
            verbose=0,
            min_char=4
        )
        self.random_ch_au = nac.RandomCharAug(
            name='Random_Char_Aug',
            action='substitute',
            aug_char_p=self.aug_p,
            aug_char_min=1,
            aug_char_max=10,
            aug_word_p=self.del_probs,
            include_numeric=False,
            include_lower_case=True,
            min_char=4,
            swap_mode='adjacent',
            spec_char='!@#$%^&*()_+',
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

    def key_keyboard_aug(self, entity, cols_aug):
        entity_aug = {}

        for key, val in entity.items():
            if key in cols_aug:
                key = str(key)
                aug_key = self.keyboard_aug.augment(key)
                if isinstance(aug_key, list):
                    aug_key = ''.join(aug_key)
                entity_aug[aug_key] = val
            else:
                entity_aug[key] = str(val)

        return entity_aug

    def val_keyboard_aug(self, entity, cols_aug):
        entity_aug = {}

        for key, val in entity.items():
            if key in cols_aug:
                val = str(val)
                aug_val = self.keyboard_aug.augment(val)
                if isinstance(aug_val, list):
                    aug_val = ''.join(aug_val)
                entity_aug[key] = aug_val
            else:
                entity_aug[key] = str(val)

        return entity_aug

    def key_random_char_aug(self, entity, cols_aug):
        entity_aug = {}

        for key, val in entity.items():
            if key in cols_aug:
                key = str(key)
                aug_key = self.random_ch_au.augment(key)
                if isinstance(aug_key, list):
                    aug_key = ''.join(aug_key)
                entity_aug[aug_key] = val
            else:
                entity_aug[key] = str(val)

        return entity_aug

    def val_random_char_aug(self, entity, cols_aug):
        entity_aug = {}

        for key, val in entity.items():
            if key in cols_aug:
                val = str(val)
                aug_val = self.random_ch_au.augment(val)
                if isinstance(aug_val, list):
                    aug_val = ''.join(aug_val)
                entity_aug[key] = aug_val
            else:
                entity_aug[key] = str(val)

        return entity_aug

    def keyboard_error_aug(self,
                           entity,
                           aug_probs_key,
                           aug_char_key_val):
        col_perc = (aug_probs_key * len(entity.keys()))
        col_perc = math.ceil(col_perc)
        columns = list(set(entity.keys()))

        cols_aug = random.choices(columns, k=col_perc)

        if aug_char_key_val == "KEY":
            entity_aug = self.key_keyboard_aug(entity, cols_aug)
        elif aug_char_key_val == "VAL":
            entity_aug = self.val_keyboard_aug(entity, cols_aug)
        elif aug_char_key_val == "BOTH":
            entity_aug = self.key_keyboard_aug(entity, cols_aug)

            col_perc = (aug_probs_key * len(entity_aug.keys()))
            col_perc = math.ceil(col_perc)
            columns = list(set(entity_aug.keys()))

            cols_aug = random.choices(columns, k=col_perc)
            entity_aug = self.val_keyboard_aug(entity_aug, cols_aug)
        else:
            entity_aug = entity

        return entity_aug

    def random_error_aug(self,
                         entity,
                         aug_probs_key,
                         aug_char_key_val):
        col_perc = (aug_probs_key * len(entity.keys()))
        col_perc = math.ceil(col_perc)
        columns = list(set(entity.keys()))

        cols_aug = random.choices(columns, k=col_perc)

        if aug_char_key_val == "KEY":
            entity_aug = self.key_random_char_aug(entity, cols_aug)
        elif aug_char_key_val == "VAL":
            entity_aug = self.val_random_char_aug(entity, cols_aug)
        elif aug_char_key_val == "BOTH":
            entity_aug = self.key_random_char_aug(entity, cols_aug)

            col_perc = (aug_probs_key * len(entity_aug.keys()))
            col_perc = math.ceil(col_perc)
            columns = list(set(entity_aug.keys()))

            cols_aug = random.choices(columns, k=col_perc)

            entity_aug = self.val_random_char_aug(entity, cols_aug)
        else:
            entity_aug = entity

        return entity_aug

    def char_augmentation(self,
                          entity,
                          aug_char_type,
                          aug_probs_key,
                          aug_char_key_val):
        if aug_char_type == "KBE":
            entity_aug = self.keyboard_error_aug(entity,
                                                 aug_probs_key,
                                                 aug_char_key_val)
        elif aug_char_type == "RCE":
            entity_aug = self.random_error_aug(entity,
                                               aug_probs_key,
                                               aug_char_key_val)
        elif aug_char_type == "BOTH":
            entity_aug = self.keyboard_error_aug(entity,
                                                 aug_probs_key,
                                                 aug_char_key_val)

            entity_aug = self.random_error_aug(entity_aug,
                                               aug_probs_key,
                                               aug_char_key_val)
        else:
            entity_aug = entity

        return entity_aug

    def execute_char_aug(self,
                         entity,
                         del_ele,
                         del_probs,
                         aug_char_type,
                         aug_probs_key,
                         aug_char_key_val):
        entities = [entity['entity_a'], entity['entity_b']]
        aug_entities = []
        for ent in entities:
            if del_ele:
                ent = self.remove_key(ent, del_probs)
            ent_af_aug = self.char_augmentation(ent,
                                                aug_char_type,
                                                aug_probs_key,
                                                aug_char_key_val)
            aug_entities.append(ent_af_aug)

        entity['entity_a'] = aug_entities[0]
        entity['entity_b'] = aug_entities[1]

        return entity


if __name__ == "__main__":
    ca = CharAugmentor()
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
    print(ca.execute_char_aug(entity))
