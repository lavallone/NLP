import re
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from spacy.cli.download import download as spacy_download
import spacy
from spacy.tokens import Doc

######################################### UTILITY PREPROCESSING FUNCTIONS ##############################################
## CLEAN TOKENS
def special_clean(data):
    # aux function
    def find_sub_list(sl,l):
        sll=len(sl)
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
                return ind, ind+sll-1
            
    regex = re.compile("<[^>]*>")
    for sample in data:
        sentence = " ".join(sample)
        list_of_matches = re.findall(regex, sentence)
        
        if list_of_matches != []:
            for elem in [e.split(" ") for e in list_of_matches]:
                s_i, e_i = find_sub_list(elem, sample)
                for i in range(s_i , e_i+1):
                    sample[i] = "<IGNORE>"

def clean_tokens(data):
    punct2tok = {"=" : "<IGNORE>", "+" : "<IGNORE>", "-" : "<IGNORE>", "_" : "<IGNORE>", "/" : "<IGNORE>", "{" : "<IGNORE>", "}" : "<IGNORE>", "@" : "<IGNORE>", "#" : "<IGNORE>", "*" : "<IGNORE>", "'" : "<IGNORE>", "<" : "<IGNORE>", ">" : "<IGNORE>"}
    puncts_list = [".", ",", ";", ":", "?", "!", "[", "]","(", ")", "$", "%", "&", "'", "=", "+", "_", "/", "{", "}", "@", "#", "*", "<", ">"] # is quite full of "-" in the middle of tokens --> so we don't delete it!
    num2digit = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'}

    special_clean(data)
    
    for sample in data:
        for i in range(len(sample)):
            sample[i] = sample[i].lower()
            sample[i] = sample[i].replace(" ", "")
            sample[i] = sample[i].replace("`", "'")
            sample[i] = sample[i].replace("[", "(")
            sample[i] = sample[i].replace("]", ")")
            sample[i] = sample[i].encode("ascii", "ignore").decode()
            if len(sample[i]) == 1 and sample[i]=="&":
                sample[i] = sample[i].replace("&", "and")
    
    # remove punctuactions/symbols and/or modify them in special tokens
    for sample in data:
        for i in range(len(sample)):
            if len(sample[i]) > 1:
                if sample[i] == "<IGNORE>" or sample[i] == "''" or sample[i] == "'s":
                    continue
                for p in puncts_list:
                    sample[i] = sample[i].replace(p, "")
            else:
                if sample[i] in list(punct2tok.keys()):
                    sample[i] = punct2tok[sample[i]]
                    
    # numbers
    for sample in data:
        for i in range(len(sample)):
            if sample[i].isnumeric():
                if len(sample[i]) > 1:
                    sample[i] = "<NUMBER>"
                else:
                    sample[i] = num2digit[sample[i]]
                    
    # after the cleaning there are probably some "empty tokens"
    # we substitute them with the <IGNORE> special token!
    for sample in data:
        for i in range(len(sample)):
            if sample[i] == "":
                sample[i] = "<IGNORE>"

## FILTER SENTENCES
def filter_sentences(train_sentences, train_labels, word2id, min_sent_length=2, max_sent_length=60):
    data_train_list = list(zip(train_sentences, train_labels))
    data_train_list = list(filter(lambda x: len(x[0])>min_sent_length and len(x[0])<max_sent_length, data_train_list)) # min and max length
    ris1, ris2 = [], []
    for e in data_train_list:
        if all((t=="unk" or t not in word2id) or l=="O" for t,l in zip(e[0], e[1])): # OOV events or all 'O' labels
            continue
        ris1.append(e[0])
        ris2.append(e[1])
    return ris1, ris2

# POS Tagging process
def pos_tagger(train_sentences):
    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("\n__________ STARTING TO EXTRACT POS TAGS __________\n")
    pos_sentences = []
    for sent in tqdm(train_sentences):
        doc = Doc(nlp.vocab, sent)
        pos_l = []
        for token in nlp(doc):
            pos_l.append(token.pos_)
        pos_sentences.append(pos_l)
    return pos_sentences

########################################################################################################################

# I took inspirantion from Notebook #4 - POS tagging
class EventDetDataset(Dataset):
    # static objects
    label2id={"B-SENTIMENT": 0, "I-SENTIMENT": 1, "B-SCENARIO": 2, "I-SCENARIO": 3, "B-CHANGE": 4, "I-CHANGE": 5,
              "B-POSSESSION": 6, "I-POSSESSION": 7, "B-ACTION": 8, "I-ACTION": 9, "O": 10}
    id2label={0 : "B-SENTIMENT", 1 : "I-SENTIMENT", 2 : "B-SCENARIO", 3 : "I-SCENARIO", 4 : "B-CHANGE", 5 : "I-CHANGE",
              6 : "B-POSSESSION", 7 : "I-POSSESSION", 8 : "B-ACTION", 9 : "I-ACTION", 10 : "O"}

    pos2id = {'ADJ': 0, 'ADP': 1, 'ADV': 2, 'AUX': 3, 'CONJ': 4, 'CCONJ': 5, 'DET': 6, 'INTJ': 7, 'NOUN': 8, 'NUM': 9, 
              'PART': 10, 'PRON': 11, 'PROPN': 12, 'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16, 'X': 17, 'SPACE': 18}

    def __init__(self, tokens, labels, pos_tokens, vocab, hparams, pred=False):

        # if 'pred'=True it means we are performing inference and we do not need labels!
        assert (pred==True and labels is None) or (pred==False and labels is not None)
        # we assume the both the tokens and the labels have been already preprocessed
        self.tokens = tokens
        self.labels = labels
        self.pos_tokens = pos_tokens
        self.window_size = hparams.window_size
        self.window_shift = hparams.window_shift
        assert self.window_shift <= self.window_size and self.window_shift >= self.window_size/2
        self.vocab = vocab
        self.windows_each_sentence_list = []
        
        tokens = self.create_windows(self.tokens)
        self.encode_text(tokens, self.vocab, labels=False)
        if pos_tokens is not None:
            pos_tokens = self.create_windows(self.pos_tokens)
            self.encode_text(pos_tokens, self.vocab, labels=False, pos=True)
        if not pred:
            self.windows_each_sentence_list = []
            labels = self.create_windows(self.labels)
            self.encode_text(labels, self.vocab, labels=True)
        
        self.data = self.make_data(tokens, labels, pos_tokens)
            

    # in a "PytorchLightning fashion" I called it 'make_data' :)
    # it embodies the "slicing windows mechanism" and the encoding phase
    def make_data(self, tokens, labels, pos_tokens):
        data = []
        if labels is None:
            if pos_tokens is None:
                for t in tokens:
                    data.append({"inputs" : t})
            else:
                for t,p in zip(tokens, pos_tokens):
                    data.append({"inputs" : t, "pos" : p})
        else:
            if pos_tokens is None:
                for t,l in zip(tokens, labels):
                    data.append({"inputs" : t, "labels" : l})
            else:
                for tl,p in zip(list(zip(tokens, labels)), pos_tokens):
                    data.append({"inputs" : tl[0], "labels" : tl[1], "pos" : p})
        return data

    def create_windows(self, data):
        ris = []
        for sample in data:
            windows_each_sentence = 0
            for i in range(0, len(sample), self.window_shift):
                windows_each_sentence += 1
                win_sample = sample[i:i+self.window_size]
                if len(win_sample) < self.window_size:
                   win_sample = win_sample + [None]*(self.window_size - len(win_sample))
                   ris.append(win_sample)
                   break
                ris.append(win_sample)
            self.windows_each_sentence_list.append(windows_each_sentence) # I'm going to need it during prediction phase
        return ris

    def encode_text(self, data, vocab, labels, pos=False):
        for sample in data:
            for i in range(len(sample)): # it should be window_size long
                if labels: # we are encoding labels
                    if sample[i] is None:
                        sample[i] = vocab["<PAD>"]
                    else: 
                        sample[i] = EventDetDataset.label2id[sample[i]]
                elif pos: # we are encoding pos tags
                    if sample[i] is None:
                        sample[i] = len(EventDetDataset.pos2id)
                    else:
                        sample[i] = EventDetDataset.pos2id[sample[i]]
                else: # we are encoding tokens
                    if sample[i] is None:
                        sample[i] = vocab["<PAD>"]
                    elif sample[i] == "unk" or sample[i] not in vocab:
                        sample[i] = vocab["<UNK>"]
                    else:
                        sample[i] = vocab[sample[i]]
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @staticmethod
    def decode_output(outputs): # outputs is a list
        decode = list()
        for i in outputs:
            decode.append(EventDetDataset.id2label[i])
        return decode
    
    @staticmethod
    # I also define the collate function (as a static one) for the DataLoader when it iterates over batches
    def collate_batch(batch):
        batch_out = dict()
        batch_out["inputs"] = torch.as_tensor([sample["inputs"] for sample in batch])
        if "labels" in list(batch[0].keys()):
            batch_out["labels"] = torch.as_tensor([sample["labels"] for sample in batch])
        if "pos" in list(batch[0].keys()):
            batch_out["pos"] = torch.as_tensor([sample["pos"] for sample in batch])
        return batch_out