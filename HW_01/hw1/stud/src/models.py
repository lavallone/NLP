from torch import nn
import torch
import json
import random
import math
from torch.autograd import Variable
from .data import EventDetDataset

def overlap_preds(l1, l2):
    ris = []
    for e1,e2 in list(zip(l1,l2)):
        if e1==e2:
            ris.append(e1)
        else:
            if random.randint(0, 1) == 0:
                ris.append(e1)
            else:
                ris.append(e2)
    return ris+[None for _ in range(len(ris))] 

# utility function for manipulating the output predictions when varying the windows size/shift!
def manipulate_preds(window_size, window_shift, all_flatten_preds, num_windows_each_sent_list):
    overlap = window_size - window_shift
    current_idx = 0
    for num_windows_each_sent in num_windows_each_sent_list:
        for i in range(num_windows_each_sent-1):
            middle = current_idx + ( window_size*(i+1) )
            start = middle-overlap
            end = middle+overlap
            all_flatten_preds[start:end] = overlap_preds(all_flatten_preds[start:middle], all_flatten_preds[middle:end])
        current_idx += (num_windows_each_sent*window_size) 
    return [e for e in all_flatten_preds if e is not None], [e-1 for e in num_windows_each_sent_list]

# this is the core function for making predictions
def predict_function(model, device, sentences, data_loader, num_windows_each_sent_list):
    # for being able to deal with sentences that have been split by the "sliding window mechanism"
    # 1) we keep all the  predictions made by model in a single flattened list...
    all_flatten_preds = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['inputs'].to(device)
            pos_inputs = None
            if model.hparams.POS_emb:
                pos_inputs = batch['pos'].to(device)
            outputs = model(inputs, pos_inputs)
            preds = torch.argmax(outputs, -1).tolist()
            all_flatten_preds += [label for sent_labels in preds for label in sent_labels]
        
        sentence_lenghts = [len(t) for t in sentences]
        all_flatten_preds, n_list = manipulate_preds(model.hparams.window_size, model.hparams.window_shift, all_flatten_preds, num_windows_each_sent_list)
        all_flatten_preds = EventDetDataset.decode_output(all_flatten_preds)
        
        # 2) then, based on the original sentences length, we recover all the predicted labels
        pred = []
        for sentence_len, n in list(zip(sentence_lenghts, n_list)):
            pred_sentence = []
            for _ in range(sentence_len):
                pred_sentence.append(all_flatten_preds.pop(0))
            pred.append(pred_sentence)
            delete_pad_idx = sentence_len + (n*(model.hparams.window_size-model.hparams.window_shift))
            # we remove from the list all the <PAD> tokens
            if delete_pad_idx % model.hparams.window_size != 0: # (the only case when we don't have to clean up the sentence)
                for _ in range(model.hparams.window_size - (delete_pad_idx%model.hparams.window_size)):
                    all_flatten_preds.pop(0)
        assert all_flatten_preds==[] # if we pass this test it means we are predicting in the right way!
    return pred


# code entirely taken (slightly changed) from the article
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#d554
class PositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_len=50, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_len, emb_dim)
        for pos in range(max_len):
            for i in range(0, emb_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/emb_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/emb_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.emb_dim)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

# I took inspiration from the "POSTaggerModel" model from NLP Notebook #4 - POS tagging 
# as a starting point for my implementation
class EventDetModel(nn.Module):
    def __init__(self, hparams):
        super(EventDetModel, self).__init__()
        self.hparams = hparams
        self.vocab = json.load(open(self.hparams.prefix_path+self.hparams.vocab_path, "r"))
        self.vocab_size = len(self.vocab)
        
        if self.hparams.num_emb==1:
            self.embedding_layer = nn.Embedding(self.vocab_size, self.hparams.emb_dim)
            if self.hparams.load_pretrained_emb:
                self.embedding_layer.load_state_dict(torch.load(self.hparams.prefix_path+self.hparams.emb_folder+str(self.hparams.emb_dim)+"/embedding_layer.pth"))
        else:
            self.embedding_layer_1 = nn.Embedding(400000, self.hparams.emb_dim)
            self.embedding_layer_2 = nn.Embedding(self.vocab_size-400000, self.hparams.emb_dim)
            if self.hparams.load_pretrained_emb:
                self.embedding_layer_1.load_state_dict(torch.load(self.hparams.prefix_path+self.hparams.emb_folder+str(self.hparams.emb_dim)+"/embedding_layer_1.pth"))
                self.embedding_layer_2.load_state_dict(torch.load(self.hparams.prefix_path+self.hparams.emb_folder+str(self.hparams.emb_dim)+"/embedding_layer_2.pth"))
        
        if not self.hparams.finetune_emb:
            if self.hparams.num_emb==1:
                for param in self.embedding_layer.parameters():
                    param.requires_grad = False
            else:
                for param in self.embedding_layer_1.parameters():
                    param.requires_grad = False 
                for param in self.embedding_layer_2.parameters():
                    param.requires_grad = False

        if self.hparams.positional_encode:
            self.positional_enc = PositionalEncoder(self.hparams.emb_dim)
        
        self.lstm = nn.LSTM(self.hparams.emb_dim, self.hparams.hidden_dim, 
                            bidirectional=self.hparams.bidirectional,
                            num_layers=self.hparams.num_layers, 
                            dropout = self.hparams.dropout if hparams.num_layers > 1 else 0,
                            batch_first=True,)
        lstm_output_dim = self.hparams.hidden_dim if self.hparams.bidirectional is False else self.hparams.hidden_dim * 2
        
        if self.hparams.POS_emb:
            self.POS_emb = nn.Embedding(len(EventDetDataset.pos2id)+1, lstm_output_dim)
            self.combination_layer = nn.Linear(lstm_output_dim, lstm_output_dim)
        
        self.dropout = nn.Dropout(self.hparams.dropout)
        if self.hparams.mlp: # multi-layer perceptron classifier
            self.classifier = nn.Sequential(
			nn.Linear(lstm_output_dim, lstm_output_dim//2),
			nn.ReLU(inplace=True),
			nn.Dropout(hparams.dropout),

			nn.Linear(lstm_output_dim//2, lstm_output_dim//4),
			nn.ReLU(inplace=True),
			nn.Dropout(hparams.dropout),

			nn.Linear(lstm_output_dim//4, self.hparams.num_classes),
			nn.ReLU(inplace=True)
		)
        else:
            self.classifier = nn.Linear(lstm_output_dim, self.hparams.num_classes)

    def forward(self, x, POS_x):
        if self.hparams.num_emb==1:
            emb = self.embedding_layer(x)
        else: # logic to follow if we use two different embedding layers
            mask = x >= 400000
            x1 = x.clone()
            x1[mask] = 0
            emb_1 = self.embedding_layer_1(x1)
            x[~mask] = 0
            x[mask] -= 400000
            emb_2 = self.embedding_layer_2(x)
            emb_1[mask] = emb_2[mask]
            emb = emb_1
            del x1
        
        if self.hparams.positional_encode:
            emb = self.positional_enc(emb) # dropout already included
        else:
            emb = self.dropout(emb)
        
        o, (_, _) = self.lstm(emb)
        
        if POS_x is not None:
            POS_emb = self.POS_emb(POS_x)
            POS_emb = self.dropout(POS_emb)
            o = self.combination_layer(o + POS_emb)
        o = self.dropout(o)
            
        # while mathematically the CrossEntropyLoss takes as input the probability distributions (hence the softmax),
        # torch optimizes its computation internally and takes as input the logits instead.
        # (that's why we returned directly the logits!)
        return self.classifier(o)