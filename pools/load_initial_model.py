import sys
import torch
import os

sys.path.append('../')
from DLEM.model import EMModel
from config.config import get_classifier_config, get_deep_learning_config


class LoadInitModel:
    def __init__(self):
        self.classifier_config = get_classifier_config()
        self.dl_config = get_deep_learning_config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_init_model(self):
        model = EMModel(self.device, self.dl_config["lm"])
        chck_path_1 = os.path.join(self.classifier_config["model_path_1"], "model.pt")
        chck_path_2 = os.path.join(self.classifier_config["model_path_2"], "model.pt")

        check_point = {
            "model": model.state_dict()
        }
        torch.save(check_point, chck_path_1)
        torch.save(check_point, chck_path_2)


def load_init_model():
    load_model = LoadInitModel()
    load_model.load_init_model()


if __name__ == "__main__":
    lm = LoadInitModel()
    lm.load_init_model()
