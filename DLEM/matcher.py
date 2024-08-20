import torch
import os
import numpy as np
import random

from torch.utils import data
from scipy.special import softmax
from transformers import AutoTokenizer

from model import EMModel
from exceptions import ModelNotFoundError
from config.config import get_deep_learning_config

dl_config = get_deep_learning_config()


def get_tokenizer(lm):
    return AutoTokenizer.from_pretrained(lm)


class EM_Matcher_Dataset(data.Dataset):
    def __init__(self, entities, max_len=128, lm='roberta-base'):
        self.pairs = []
        self.max_len = max_len
        self.tokenizer = get_tokenizer(lm)

        for entity in entities:
            entity_a, entity_b = entity.strip().split('\t')
            self.pairs.append((entity_a, entity_b))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        entity_a = self.pairs[item][0]
        entity_b = self.pairs[item][1]

        x = self.tokenizer.encode(
            text=entity_a,
            text_pair=entity_b,
            max_length=self.max_len,
            padding='longest',
            truncation=True
        )

        return torch.LongTensor(x)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def classify(
        entity_pairs,
        model,
        lm='distilbert',
        max_len=128,
        threshold=None
):
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = EM_Matcher_Dataset(
        entities=entity_pairs,
        max_len=max_len,
        lm=lm
    )

    iterator = data.DataLoader(
        dataset=dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=0
    )

    all_probs = []
    all_logits = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x = batch
            logits = model(x)
            probs = logits.softmax(dim=1)
            probs = probs[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_logits += logits.cpu().numpy().tolist()

    if threshold is None:
        threshold = 0.5

    pred = [1 if p > threshold else 0 for p in all_probs]

    print(pred, all_logits)

    return pred, all_logits


def load_model(model_path, lm):
    checkpoint = model_path

    if not os.path.exists(checkpoint):
        raise ModelNotFoundError(checkpoint)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = EMModel(
        device=device,
        lm=lm
    )

    saved_state = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(saved_state['model'])
    th = saved_state['threshold']
    model = model.to(device)

    return model, th


def serialize_entity(entity_a, entity_b, weight):
    entities = [entity_a, entity_b]
    entity_pairs = []
    ent_idx = []
    for entity in entities:
        entity_string = ""
        for key, val in entity.items():
            if key == 'idx':
                ent_idx.append(int(val))
            else:
                if val != 'nan':
                    val = val.replace('\n', ' ')
                    key = key.replace('\n', '')
                    val = val.replace('\t', ' ')
                    key = key.replace('\t', '')
                    entity_string += f"COL {key} VAL {val} "
        entity_string += f"COL WEIGHT VAL {weight}"
        entity_string = entity_string.strip()
        entity_pairs.append(entity_string)
    entity_pairs = '\t'.join(entity_pairs)

    return entity_pairs


def predict(entity_a, entity_b, weight):
    lm = dl_config['lm']
    model_path = os.path.join(dl_config["model_path"], 'model.pt')
    max_len = dl_config["MAX_LEN"]
    set_seed(123)

    model, threshold = load_model(model_path=model_path, lm=lm)

    entities = [serialize_entity(entity_a, entity_b, weight)]

    predictions, logits = classify(
        entity_pairs=entities,
        model=model,
        lm=lm,
        max_len=max_len,
        threshold=threshold

    )

    scores = softmax(logits, axis=1)

    for row, pred, score in zip(entities, predictions, scores):
        output = {'entities': row,
                  'match': pred,
                  'match_confidence': score[int(pred)]}
        print(output)


if __name__ == "__main__":
    entity_a = {
        "idx": 1535,
        "profile_id": "Match12900",
        "editor": "nan",
        "starring": "Aidan Murphy",
        "title": "nan",
        "writer": "Patrick Chapman"
    }

    entity_b = {
        "idx": 1535,
        "profile_id": "Match21361",
        "year": "1985",
        "actor_name": "Murphy, Charles Thomas",
        "director_name": "Clement, Dick",
        "genre": "Adventure",
        "imdb_ksearch_id": "835624",
        "title": "Water (1985/I)",
        "url": "http://imdb.com/title/tt0090297"
    }

    weight = 2.0

    predict(entity_a, entity_b, weight)
