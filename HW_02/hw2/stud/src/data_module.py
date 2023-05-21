import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import json
from transformers import BertTokenizerFast, RobertaTokenizerFast, DebertaTokenizerFast
from .utils import read_dataset

######################################### UTILITY PREPROCESSING FUNCTIONS ##############################################
## CLEAN TOKENS
def clean_tokens(data): # very simple token cleaner (not needed to make big operations here because BERT encoder works pretty well!)
    for sample in data:
        for i in range(len(sample["words"])):
            sample["words"][i] = sample["words"][i].lower()
            sample["words"][i] = sample["words"][i].replace(" ", "")
            sample["words"][i] = sample["words"][i].replace("`", "'")
            sample["words"][i] = sample["words"][i].replace("[", "(")
            sample["words"][i] = sample["words"][i].replace("]", ")")
            sample["words"][i] = sample["words"][i].encode("ascii", "ignore").decode()
        for i in range(len(sample["lemmas"])): # do it also for lemmas!
            sample["lemmas"][i] = sample["lemmas"][i].lower()
            sample["lemmas"][i] = sample["lemmas"][i].replace(" ", "")
            sample["lemmas"][i] = sample["lemmas"][i].replace("`", "'")
            sample["lemmas"][i] = sample["lemmas"][i].replace("[", "(")
            sample["lemmas"][i] = sample["lemmas"][i].replace("]", ")")
            sample["lemmas"][i] = sample["lemmas"][i].encode("ascii", "ignore").decode()
                
## FILTER SENTENCES
def filter_sentences(train_items, min_sent_length=5, max_sent_length=85):
    train_items = list(filter(lambda x: len(x["words"])>=min_sent_length and len(x["words"])<=max_sent_length, train_items)) # min and max length
    for item in train_items:
        # we check that each train sentence has at least one word to be disambiguated!
        assert len(item["instance_ids"].keys()) != 0
    return train_items

## MAPPING BETWEEN INPUT WORD INDEX AND BERT EMBEDDING INDECES
# (I also want to retrieve the index of the POS TAG)
def token2emb_idx(sense_idx, word_ids, use_POS=False):
    ris = []
    i = 0
    for word_id in word_ids:
        if ris != [] and word_id != sense_idx: # to make it more efficient
            break
        if word_id==sense_idx:
            ris.append(i)
        i+=1
    if use_POS: # we append the POS tag index at the end of the list
        i=1
        for word_id in word_ids[1:]:
            if word_id is None:
                i+=1
                break
            i+=1
        ris.append(i)        
    return ris
     
########################################################################################################################                

class CoarseWSD_Dataset(Dataset):
    def __init__(self, data_sentences, data_senses, sense2id_path, use_lemmas=False, use_POS=False):
        self.data = list()
        self.data_sentences = data_sentences
        self.data_senses = data_senses
        self.sense2id_path = sense2id_path
        self.use_lemmas = use_lemmas
        self.use_POS = use_POS
        self.make_data()
    
    def make_data(self):
        for i,d in enumerate(self.data_sentences):
            if self.data_senses is None: # we are predicting
                for sense_idx in d["instance_ids"].keys():
                    sense_idx = int(sense_idx)
                    if self.use_lemmas:
                        sentence = " ".join(d["lemmas"])
                    else:
                        sentence = " ".join(d["words"])
                    if self.use_POS:
                        input_sentence = [sentence, d["pos_tags"][sense_idx].lower()]
                    else:
                        input_sentence = sentence
                    
                    # mapping between senses and their respective indeces
                    sense2id = json.load(open(self.sense2id_path, "r"))
                    candidates = []
                    for c in d["candidates"][str(sense_idx)]:
                        if c not in sense2id.keys():
                            candidates.append(sense2id["<UNK>"]) # <UNK> INDEX (2158)
                        else:
                            candidates.append(sense2id[c])
                    self.data.append({"sense_idx" : sense_idx, "input": input_sentence, "candidates" : candidates})
            
            else: # we are not predicting   
                for sense_idx, true_sense in zip(d["instance_ids"].keys(), self.data_senses[i]):
                    sense_idx = int(sense_idx)
                    if self.use_lemmas:
                        sentence = " ".join(d["lemmas"])
                    else:
                        sentence = " ".join(d["words"])
                    if self.use_POS:
                        input_sentence = [sentence, d["pos_tags"][sense_idx].lower()]
                    else:
                        input_sentence = sentence
                    
                    # mapping between senses and their respective indeces
                    sense2id = json.load(open(self.sense2id_path, "r"))
                    true_sense = sense2id[true_sense[0]]
                    candidates = []
                    for c in d["candidates"][str(sense_idx)]:
                        if c not in sense2id.keys():
                            candidates.append(sense2id["<UNK>"]) # <UNK> INDEX (2158)
                        else:
                            candidates.append(sense2id[c])
                    self.data.append({"sense_idx" : sense_idx, "input": input_sentence, "labels" : true_sense, "candidates" : candidates})
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class WSD_DataModule(pl.LightningDataModule):
    def __init__(self, hparams, is_predict=False, sentence_to_predict=None):
        super().__init__()
        self.save_hyperparameters(hparams, logger=False)
        
        self.is_predict = is_predict
        assert self.is_predict is False or sentence_to_predict is not None
        if self.is_predict is True: # we are predicting trough the test.sh script
            self.test_sentences, self.test_senses = (sentence_to_predict, None)
        else: # we want to evaluate our model with 'evaluation_pipeline()'
            self.train_sentences, self.train_senses = read_dataset(self.hparams.prefix_path+self.hparams.data_train)
            self.val_sentences, self.val_senses = read_dataset(self.hparams.prefix_path+self.hparams.data_val)
            self.test_sentences, self.test_senses = read_dataset(self.hparams.prefix_path+self.hparams.data_test)

    def setup(self, stage=None):
        if self.is_predict is True: # only prediction
            # TEST
            clean_tokens(self.test_sentences)
            self.data_test = CoarseWSD_Dataset(data_sentences=self.test_sentences, data_senses=self.test_senses, sense2id_path=self.hparams.prefix_path+"model/files/"+self.hparams.coarse_or_fine+"_sense2id.json", use_lemmas=self.hparams.use_lemmas, use_POS=self.hparams.use_POS)
        else:
            # TRAIN
            clean_tokens(self.train_sentences)
            self.data_train = CoarseWSD_Dataset(data_sentences=filter_sentences(self.train_sentences), data_senses=self.train_senses, sense2id_path=self.hparams.prefix_path+"model/files/"+self.hparams.coarse_or_fine+"_sense2id.json", use_lemmas=self.hparams.use_lemmas, use_POS=self.hparams.use_POS)
            # VAL
            clean_tokens(self.val_sentences)
            self.data_val = CoarseWSD_Dataset(data_sentences=self.val_sentences, data_senses=self.val_senses, sense2id_path=self.hparams.prefix_path+"model/files/"+self.hparams.coarse_or_fine+"_sense2id.json", use_lemmas=self.hparams.use_lemmas, use_POS=self.hparams.use_POS)
            # TEST
            clean_tokens(self.test_sentences)
            self.data_test = CoarseWSD_Dataset(data_sentences=self.test_sentences, data_senses=self.test_senses, sense2id_path=self.hparams.prefix_path+"model/files/"+self.hparams.coarse_or_fine+"_sense2id.json", use_lemmas=self.hparams.use_lemmas, use_POS=self.hparams.use_POS)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.n_cpu,
            collate_fn = self.collate,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.n_cpu,
            collate_fn = self.collate,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True
        )
        
    def test_dataloader(self):
        if self.is_predict is False:
            return DataLoader(
                self.data_test,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.hparams.n_cpu,
                collate_fn = self.collate,
                pin_memory=self.hparams.pin_memory,
                persistent_workers=True
            )
        else:
            return DataLoader(
                self.data_test,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.hparams.n_cpu,
                collate_fn = self.pred_collate,
                pin_memory=self.hparams.pin_memory,
                persistent_workers=True
            )
    
    # for efficiency reasons, each time we pick a batch from the dataloader, we call this function!
    def collate(self, batch):
        batch_out = dict()
        if self.hparams.encoder_type == "bert":
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        elif self.hparams.encoder_type == "roberta":
            tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        elif self.hparams.encoder_type == "deberta":
            tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-base")
        batch_out["input"] = tokenizer([sample["input"] for sample in batch], padding=True, truncation=True, return_tensors="pt")
        # we now map token to embedding indices
        batch_out["sense_ids"] = [token2emb_idx(batch[i]["sense_idx"], batch_out["input"].word_ids(i), self.hparams.use_POS) for i in range(len(batch))]
        batch_out["labels"] = [sample["labels"] for sample in batch]
        batch_out["candidates"] = [sample["candidates"] for sample in batch]
        return batch_out
    
    # for efficiency reasons, each time we pick a batch from the dataloader, we call this function!
    def pred_collate(self, batch):
        batch_out = dict()
        if self.hparams.encoder_type == "bert":
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        elif self.hparams.encoder_type == "roberta":
            tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        elif self.hparams.encoder_type == "deberta":
            tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-base")
        batch_out["input"] = tokenizer([sample["input"] for sample in batch], padding=True, truncation=True, return_tensors="pt")
        # we now map token to embedding indices
        batch_out["sense_ids"] = [token2emb_idx(batch[i]["sense_idx"], batch_out["input"].word_ids(i), self.hparams.use_POS) for i in range(len(batch))]
        batch_out["candidates"] = [sample["candidates"] for sample in batch]
        return batch_out