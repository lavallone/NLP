import numpy as np
from typing import List
import json
import torch
from torch.utils.data import DataLoader

from model import Model
from stud.src.hyperparameters import Hparams
from stud.src.models import EventDetModel, predict_function
from stud.src.data import EventDetDataset, clean_tokens, pos_tagger

# here I have to return the StudentModel (the best one I implemented!)
def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    hparams = Hparams()
    return StudentModel(device, hparams)


class RandomBaseline(Model):
    options = [
        (22458, "B-ACTION"),
        (13256, "B-CHANGE"),
        (2711, "B-POSSESSION"),
        (6405, "B-SCENARIO"),
        (3024, "B-SENTIMENT"),
        (457, "I-ACTION"),
        (583, "I-CHANGE"),
        (30, "I-POSSESSION"),
        (505, "I-SCENARIO"),
        (24, "I-SENTIMENT"),
        (463402, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self, device, hparams):
        self.device = device
        self.hparams = hparams
        # load vocabulary
        self.vocab = json.load(open(self.hparams.vocab_path, "r"))
        self.hparams.vocab_size = len(self.vocab)
        # load model
        self.hparams.prefix_path = ""
        self.model = EventDetModel(self.hparams).to(self.device)
        # load the weights of the model!
        self.model.load_state_dict(torch.load(self.hparams.student_weights_path, map_location=device))

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        self.hparams.window_size = 40
        self.hparams.window_shift = 40
        
        clean_tokens(tokens)
        pos_tokens = None
        if self.hparams.POS_emb:
            pos_tokens = pos_tagger(tokens)
            print("| POS TAGS extracted! |\n")
        pred_dataset = EventDetDataset(tokens, None, pos_tokens, self.vocab, self.hparams, pred=True)
        pred_dataloader = DataLoader(pred_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.n_cpu, pin_memory=self.hparams.pin_memory, collate_fn=EventDetDataset.collate_batch)
        
        return predict_function(self.model, self.device, tokens, pred_dataloader, pred_dataset.windows_each_sentence_list)