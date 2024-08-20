import datetime
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from classifier.entity_pair_classifier import EntityPairClassifier
from config.config import get_deep_learning_config, get_classifier_config

lm_mp = {
    'roberta': 'roberta-base',
    'distilbert': 'distilbert-base-uncased'
}


class Model:
    def __init__(self):
        self.classifier_config = get_classifier_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dl_config = get_deep_learning_config()
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[self.dl_config["lm"]], return_dict=False)

        classifier = EntityPairClassifier()
        pre_trained_model = os.path.join(self.classifier_config["model_path_2"], 'model.pt')
        saved_state = torch.load(pre_trained_model, map_location=torch.device(self.device))

        classifier.load_state_dict(saved_state['model'])
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

        print("\nFlask server started...")
        print(f"The model2 is read at {datetime.datetime.now()}")

    def predict(self, text):
        entity_a, entity_b = text.strip().split("\t")

        # entity A
        e_a = self.tokenizer.encode(text=entity_a,
                                    max_length=self.dl_config["MAX_LEN"],
                                    truncation=True,
                                    padding='max_length',
                                    return_tensors="pt")
        # entity B
        e_b = self.tokenizer.encode(text=entity_b,
                                    max_length=self.dl_config["MAX_LEN"],
                                    truncation=True,
                                    padding='max_length',
                                    return_tensors="pt")
        # entity A + entity B
        e_ab = self.tokenizer.encode(text=entity_a,
                                    text_pair=entity_b,
                                    max_length=self.dl_config["MAX_LEN"],
                                    truncation=True,
                                    padding='max_length',
                                    return_tensors="pt")

        e_a_tensors = e_a.to(self.device)
        e_b_tensors = e_b.to(self.device)
        e_ab_tensors = e_ab.to(self.device)

        with torch.no_grad():
            probabilities = F.softmax(self.classifier(e_a_tensors, e_b_tensors,
                                                       e_ab_tensors), dim=1)

        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()

        return (
            predicted_class,
            confidence
        )


model = Model()


def get_model2():
    return model
