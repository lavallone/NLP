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
def filter_sentences(train_sentences, train_senses, min_sent_length=5, max_sent_length=85):
    train_items = list(zip(train_sentences, train_senses))
    train_items = list(filter(lambda x: len(x[0]["words"])>=min_sent_length and len(x[0]["words"])<=max_sent_length, train_items)) # min and max length
    for item in train_items:
        # we check that each train sentence has at least one word to be disambiguated!
        assert len(item[0]["instance_ids"].keys()) != 0
    ris1, ris2 = [], []
    for e1,e2 in train_items:
        ris1.append(e1)
        ris2.append(e2)
    return ris1, ris2

## MAPPING BETWEEN INPUT WORD INDEX AND BERT EMBEDDING INDECES
## (needed after the encoding part to combine the embeddings relative to the same input token!)
def token2emb_idx(sense_idx, word_ids):
    ris = []
    i = 0
    for word_id in word_ids:
        if ris != [] and word_id != sense_idx: # to make it more efficient
            break
        if word_id==sense_idx:
            ris.append(i)
        i+=1       
    return ris
     
########################################################################################################################                

class WSD_Dataset(Dataset):
    def __init__(self, data_sentences, data_senses, sense2id_path, use_lemmas=False, use_POS=False):
        self.data = list()
        self.data_sentences = data_sentences
        self.data_senses = data_senses
        self.sense2id = json.load(open(sense2id_path, "r"))
        self.use_lemmas = use_lemmas
        self.use_POS = use_POS
        self.make_data()
    
    def make_data(self):
        for i,d in enumerate(self.data_sentences):
            # only PREDICTION
            if self.data_senses is None: 
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
                    
                    candidates = []
                    for c in d["candidates"][str(sense_idx)]:
                        # ___________________ <UNK> senses handling ________________
                        if c not in self.sense2id.keys():
                            candidates.append(self.sense2id["<UNK>"]) # <UNK> INDEX
                        # __________________________________________________________
                        else:
                            candidates.append(self.sense2id[c])
                    self.data.append({"sense_idx" : sense_idx, "input": input_sentence, "candidates" : candidates})
            
            else: # we also have LABELS
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
                    
                    true_sense = self.sense2id[true_sense[0]]
                    candidates = []
                    for c in d["candidates"][str(sense_idx)]:
                        # ___________________ <UNK> senses handling ________________
                        if c not in self.sense2id.keys():
                            candidates.append(self.sense2id["<UNK>"]) # <UNK> INDEX
                        # __________________________________________________________
                        else:
                            candidates.append(self.sense2id[c])
                    self.data.append({"sense_idx" : sense_idx, "input": input_sentence, "label" : true_sense, "candidates" : candidates})
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class WSD_Gloss_Dataset(Dataset):
    def __init__(self, data_sentences, data_senses, coarse_or_fine, sense2id_path, fine2coarse_path, sense_map_path):
        self.data = list()
        self.data_sentences = data_sentences
        self.data_senses = data_senses
        self.coarse_or_fine = coarse_or_fine
        self.sense2id = json.load(open(sense2id_path, "r"))
        self.fine2coarse = json.load(open(fine2coarse_path, "r"))
        self.sense_map = json.load(open(sense_map_path, "r"))
        self.make_data()
    
    def make_data(self):
        for i,d in enumerate(self.data_sentences):
            # only PREDICTION
            if self.data_senses is None:
                for sense_idx, id in d["instance_ids"].items():
                    sense_idx = int(sense_idx)
                    context_sentence = " ".join(d["words"])
                    
                    # _________________________________ <UNK> senses handling _________________________________
                    is_unk = False
                    candidates = []
                    for c in d["candidates"][str(sense_idx)]:
                        # if the candidate sense is not in the sense inventory, we load none of the samples!
                        if c not in self.sense2id.keys():
                            self.data.append({"id" : id, "synset" : "<UNK>", "sense_idx" : 0, "input": "UNK"})
                            is_unk = True
                            break
                        else:
                            candidates.append(c)
                    # if we find at least one candidate that is not in the inventory, we exit the loop!
                    if is_unk:
                        continue
                    # _________________________________________________________________________________________
                            
                    if self.coarse_or_fine == "fine":
                        # we list all the respective coarse-grained senses
                        coarse_candidates = list(set([ self.fine2coarse[c] for c in candidates]))
                        for c in coarse_candidates:
                            for fine_dict in self.sense_map[c]:
                                fine_sense = list(fine_dict.keys())[0]
                                fine_gloss = (list(fine_dict.values())[0]).lower() # I just lower case each fine gloss
                                input_sentence = [context_sentence, fine_gloss]
                                self.data.append({"id" : id, "synset" : fine_sense, "sense_idx" : sense_idx, "input": input_sentence})

                    else: # coarse case where we need to concatenate all the fine-grained glosses (not very effective)
                        for c in candidates:
                            concatenated_glosses_list = []
                            for fine_dict in self.sense_map[c]:
                                fine_gloss = (list(fine_dict.values())[0]).lower()
                                concatenated_glosses_list.append(fine_gloss)
                            concatenated_glosses = " ".join(concatenated_glosses_list)
                            input_sentence = [context_sentence, concatenated_glosses]
                            self.data.append({"id" : id, "synset" : c, "sense_idx" : sense_idx, "input": input_sentence})
            
            else: # we also have LABELS
                for (sense_idx, id), true_sense in zip(d["instance_ids"].items(), self.data_senses[i]):
                    sense_idx = int(sense_idx)
                    context_sentence = " ".join(d["words"])
                    
                    true_sense = true_sense[0]
                    # _________________________________ <UNK> senses handling _________________________________
                    is_unk = False
                    candidates = []
                    for c in d["candidates"][str(sense_idx)]:
                        # if the candidate sense is not in the sense inventory, we load none of the samples!
                        if c not in self.sense2id.keys():
                            self.data.append({"id" : id, "synset" : "<UNK>", "sense_idx" : 0, "input": "UNK", "label" : 0})
                            is_unk = True
                            break
                        else:
                            candidates.append(c)
                    # if we find at least one candidate that is not in the inventory, we exit the loop!
                    if is_unk:
                        continue
                    # _________________________________________________________________________________________
                            
                    if self.coarse_or_fine == "fine":
                        # we list all the respective coarse-grained senses
                        coarse_candidates = list(set([ self.fine2coarse[c] for c in candidates]))
                        for c in coarse_candidates:
                            for fine_dict in self.sense_map[c]:
                                fine_sense = list(fine_dict.keys())[0]
                                fine_gloss = (list(fine_dict.values())[0]).lower()
                                input_sentence = [context_sentence, fine_gloss]
                                if fine_sense == true_sense: # it's the TRUE sense --> we append 1!
                                    self.data.append({"id" : id, "synset" : fine_sense, "sense_idx" : sense_idx, "input": input_sentence, "label" : 1})
                                else:
                                    self.data.append({"id" : id, "synset" : fine_sense, "sense_idx" : sense_idx, "input": input_sentence, "label" : 0})

                    else: # coarse case where we need to concatenate all the fine-grained glosses (not very effective)
                        for c in candidates:
                            concatenated_glosses_list = []
                            for fine_dict in self.sense_map[c]:
                                fine_gloss = (list(fine_dict.values())[0]).lower()
                                concatenated_glosses_list.append(fine_gloss)
                            concatenated_glosses = " ".join(concatenated_glosses_list)
                            input_sentence = [context_sentence, concatenated_glosses]
                            if c == true_sense: # it's the TRUE sense --> we append 1!
                                self.data.append({"id" : id, "synset" : c, "sense_idx" : sense_idx, "input": input_sentence, "label" : 1})
                            else:
                                self.data.append({"id" : id, "synset" : c, "sense_idx" : sense_idx, "input": input_sentence, "label" : 0})

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
            self.data_test = WSD_Dataset(data_sentences=self.test_sentences, data_senses=self.test_senses, sense2id_path=self.hparams.prefix_path+"model/files/"+self.hparams.coarse_or_fine+"_sense2id.json", use_lemmas=self.hparams.use_lemmas, use_POS=self.hparams.use_POS)
        else:
            # TRAIN
            clean_tokens(self.train_sentences)
            self.train_sentences, self.train_senses = filter_sentences(self.train_sentences, self.train_senses)
            self.data_train = WSD_Dataset(data_sentences=self.train_sentences, data_senses=self.train_senses, sense2id_path=self.hparams.prefix_path+"model/files/"+self.hparams.coarse_or_fine+"_sense2id.json", use_lemmas=self.hparams.use_lemmas, use_POS=self.hparams.use_POS)
            # VAL
            clean_tokens(self.val_sentences)
            self.data_val = WSD_Dataset(data_sentences=self.val_sentences, data_senses=self.val_senses, sense2id_path=self.hparams.prefix_path+"model/files/"+self.hparams.coarse_or_fine+"_sense2id.json", use_lemmas=self.hparams.use_lemmas, use_POS=self.hparams.use_POS)
            # TEST
            clean_tokens(self.test_sentences)
            self.data_test = WSD_Dataset(data_sentences=self.test_sentences, data_senses=self.test_senses, sense2id_path=self.hparams.prefix_path+"model/files/"+self.hparams.coarse_or_fine+"_sense2id.json", use_lemmas=self.hparams.use_lemmas, use_POS=self.hparams.use_POS)

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
        # notice that I used FastTokenizers because they have 'word_ids()' method which I need for the token-embedddings mapping!
        batch_out["inputs"] = tokenizer([sample["input"] for sample in batch], padding=True, truncation=True, return_tensors="pt")
        # we now map token idx to embedding indices (from sense_idx to sense_ids)
        batch_out["sense_ids"] = [token2emb_idx(batch[i]["sense_idx"], batch_out["inputs"].word_ids(i)) for i in range(len(batch))]
        batch_out["labels"] = [sample["label"] for sample in batch]
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
        # notice that I used FastTokenizers because they have 'word_ids()' method which I need for the token-embedddings mapping!
        batch_out["inputs"] = tokenizer([sample["input"] for sample in batch], padding=True, truncation=True, return_tensors="pt")
        # we now map token idx to embedding indices (from sense_idx to sense_ids)
        batch_out["sense_ids"] = [token2emb_idx(batch[i]["sense_idx"], batch_out["input"].word_ids(i)) for i in range(len(batch))]
        batch_out["candidates"] = [sample["candidates"] for sample in batch]
        return batch_out

class WSD_Gloss_DataModule(pl.LightningDataModule):
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
            self.data_test = WSD_Gloss_Dataset(data_sentences=self.test_sentences, data_senses=self.test_senses, 
                                               coarse_or_fine=self.hparams.coarse_or_fine, 
                                               sense2id_path=self.hparams.prefix_path+"model/files/"+self.hparams.coarse_or_fine+"_sense2id.json", 
                                               fine2coarse_path=self.hparams.prefix_path+"model/files/fine2coarse.json", 
                                               sense_map_path=self.hparams.prefix_path+self.hparams.sense_map)
        else:
            # TRAIN
            clean_tokens(self.train_sentences)
            self.train_sentences, self.train_senses = filter_sentences(self.train_sentences, self.train_senses)
            self.data_train = WSD_Gloss_Dataset(data_sentences=self.train_sentences, data_senses=self.train_senses, 
                                                coarse_or_fine=self.hparams.coarse_or_fine, 
                                                sense2id_path=self.hparams.prefix_path+"model/files/"+self.hparams.coarse_or_fine+"_sense2id.json", 
                                                fine2coarse_path=self.hparams.prefix_path+"model/files/fine2coarse.json", 
                                                sense_map_path=self.hparams.prefix_path+self.hparams.sense_map)
            # VAL
            clean_tokens(self.val_sentences)
            self.data_val = WSD_Gloss_Dataset(data_sentences=self.val_sentences, data_senses=self.val_senses, 
                                              coarse_or_fine=self.hparams.coarse_or_fine, 
                                              sense2id_path=self.hparams.prefix_path+"model/files/"+self.hparams.coarse_or_fine+"_sense2id.json", 
                                              fine2coarse_path=self.hparams.prefix_path+"model/files/fine2coarse.json", 
                                              sense_map_path=self.hparams.prefix_path+self.hparams.sense_map)
            # TEST
            clean_tokens(self.test_sentences)
            self.data_test = WSD_Gloss_Dataset(data_sentences=self.test_sentences, data_senses=self.test_senses, 
                                               coarse_or_fine=self.hparams.coarse_or_fine, 
                                               sense2id_path=self.hparams.prefix_path+"model/files/"+self.hparams.coarse_or_fine+"_sense2id.json", 
                                               fine2coarse_path=self.hparams.prefix_path+"model/files/fine2coarse.json", 
                                               sense_map_path=self.hparams.prefix_path+self.hparams.sense_map)

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
    
    def collate(self, batch):
        batch_out = dict()
        if self.hparams.encoder_type == "bert":
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        elif self.hparams.encoder_type == "roberta":
            tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        elif self.hparams.encoder_type == "deberta":
            tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-base")
        batch_out["ids"] = [sample["id"] for sample in batch]
        batch_out["synsets"] = [sample["synset"] for sample in batch]
        batch_out["inputs"] = tokenizer([sample["input"] for sample in batch], padding=True, truncation=True, return_tensors="pt")
        batch_out["sense_ids"] = [token2emb_idx(batch[i]["sense_idx"], batch_out["inputs"].word_ids(i)) for i in range(len(batch))]
        batch_out["labels"] = [sample["label"] for sample in batch]
        return batch_out
    
    def pred_collate(self, batch):
        batch_out = dict()
        if self.hparams.encoder_type == "bert":
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        elif self.hparams.encoder_type == "roberta":
            tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        elif self.hparams.encoder_type == "deberta":
            tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-base")
        batch_out["ids"] = [sample["id"] for sample in batch]
        batch_out["synsets"] = [sample["synset"] for sample in batch]
        batch_out["inputs"] = tokenizer([sample["input"] for sample in batch], padding=True, truncation=True, return_tensors="pt")
        batch_out["sense_ids"] = [token2emb_idx(batch[i]["sense_idx"], batch_out["inputs"].word_ids(i)) for i in range(len(batch))]
        return batch_out