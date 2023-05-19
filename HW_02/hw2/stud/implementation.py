import numpy as np
from typing import List, Dict

from model import Model
from stud.src.hyperparameters import Hparams
from stud.src.models import WSD_Model
from stud.src.data_module import WSD_DataModule
import torch
import json


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    hparams = Hparams()
    return StudentModel(device, hparams)


class RandomBaseline(Model):

    def __init__(self):
        # Load your models/tokenizer/etc. that only needs to be loaded once when doing inference
        pass

    def predict(self, sentences: List[Dict]) -> List[List[str]]:
        return [[np.random.choice(candidates) for candidates in sentence_data["candidates"].values()]
                for sentence_data in sentences]


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self, device, hparams):
        # Load your models/tokenizer/etc. that only needs to be loaded once when doing inference
        self.device = device
        self.hparams = hparams
        # load model and its weights!
        self.hparams.prefix_path = ""
        self.model = WSD_Model.load_from_checkpoint(self.hparams.student_weights_path, strict=False, device=self.device)

    def predict(self, sentences: List[Dict]) -> List[List[str]]: # I changed from List[List[str]] to List[Dict] because our dataset is a list of dictionaries!
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        data = WSD_DataModule(self.model.hparams, is_predict=True, sentence_to_predict=sentences)
        data.setup()
        self.model.eval()
        with torch.no_grad():
            preds_list = []
            for batch in data.test_dataloader():
                preds = self.model.predict(batch)
                preds_list += preds
                
            # 1) let's first decode the predicted senses from indices to strings
            id2sense = json.load(open(self.hparams.prefix_path+"model/files/id2sense.json", "r"))
            for i in range(len(preds_list)):
                preds_list[i] = id2sense[str(preds_list[i])]
            
            # 2) now I have a unique list of predictions --> I need it to create a  list of lists where each
            #                                                sublist contains the sense of the words of the same sentence!
            senses_each_sentence = []
            for sent in sentences:
                senses_each_sentence.append(len(sent["instance_ids"].keys()))
            tot_senses = 0
            for e in senses_each_sentence:
                tot_senses += e
            assert tot_senses == len(preds_list)
            
            start_index_sense_list = []
            i=0
            for e in senses_each_sentence:
                start_index_sense_list.append(i)
                i+=e
            
            final_preds_list = []
            for e1,e2 in zip(start_index_sense_list, senses_each_sentence):
                final_preds_list.append(preds_list[e1:e2])
            
        return final_preds_list