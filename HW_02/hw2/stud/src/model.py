import torch
from torch import optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from transformers import BertModel, RobertaModel, DebertaModel
from torchmetrics import F1Score
import json

class WSD_Model(pl.LightningModule):
    def __init__(self, hparams, coarse_filter_model=None):
        super(WSD_Model, self).__init__()
        self.save_hyperparameters(hparams)
        if self.hparams.encoder_type == "bert":
            self.encoder = BertModel.from_pretrained("bert-base-uncased")
        elif self.hparams.encoder_type == "roberta":
            self.encoder = RobertaModel.from_pretrained("roberta-base")
        elif self.hparams.encoder_type == "deberta":
            self.encoder = DebertaModel.from_pretrained("microsoft/deberta-base")
        
        # we set all parameters to be not trainable
        for param in self.encoder.parameters():
            param.requires_grad = False
        # here we decide which parameters unfreeze
        if self.hparams.fine_tune_bert is True:
            if self.hparams.encoder_type == "bert":
                unfreeze = [6,7,8,9,10,11] # (if we unfreeze more layers it is not guaranteed that the perforomance will improve!)
            elif self.hparams.encoder_type == "roberta":
                unfreeze = [7,8,9,10,11]
            elif self.hparams.encoder_type == "deberta":
                unfreeze = [9,10,11]
            for i in unfreeze:
                for param in self.encoder.encoder.layer[i].parameters():
                    param.requires_grad = True
        
        self.batch_norm = nn.BatchNorm1d(768)
        self.hidden_MLP = nn.Linear(768, self.hparams.hidden_dim, bias=True)
        if self.hparams.act_fun == "relu":
            self.act_fun = nn.ReLU(inplace=True)
        if self.hparams.act_fun == "silu":
            self.act_fun = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.classifier = nn.Linear(self.hparams.hidden_dim, self.hparams.num_senses, bias=False) # final linear projection with no bias
        
        self.val_micro_f1 = F1Score(task="multiclass", num_classes=self.hparams.num_senses, average="micro")
        
        if self.hparams.coarse_loss_oriented or self.hparams.predict_coarse_with_fine:
            self.fine2coarse = json.load(open(self.hparams.prefix_path+"model/files/fine2coarse.json", "r"))
            self.fine_id2sense = json.load(open(self.hparams.prefix_path+"model/files/fine_id2sense.json", "r"))
            
        if self.hparams.predict_fine_with_coarse_filter:
            self.coarse_filter_model = coarse_filter_model
            self.coarse2fine = json.load(open(self.hparams.prefix_path+self.hparams.sense_map, "r"))
            self.fine_sense2id = json.load(open(self.hparams.prefix_path+"model/files/fine_sense2id.json", "r"))
        
        self.first_sense_statistic = 0 # I count all the time the model predict the first sense (the most frequent one) without considering the sets with only one candidate!   
       
    def forward(self, batch):
        text = batch["inputs"]
        if self.hparams.encoder_type == "roberta": # roberta doesn't need "token_type_ids"
            embed_text = self.encoder(text["input_ids"], attention_mask=text["attention_mask"], output_hidden_states=True)
        else:
            embed_text = self.encoder(text["input_ids"], attention_mask=text["attention_mask"], token_type_ids=text["token_type_ids"], output_hidden_states=True)
        # I take the hidden representation of the last four layers of each token
        if self.hparams.sum_or_mean == "sum":
            embed_text = torch.stack(embed_text.hidden_states[-4:], dim=0).sum(dim=0)
        elif self.hparams.sum_or_mean == "mean":
            embed_text = torch.stack(embed_text.hidden_states[-4:], dim=0).mean(dim=0)
        
        # I select the embeddings of the word we want to disambiguate and take their average! (for each item in the batch)
        encoder_output_list = []
        for i in range(len(batch["sense_ids"])):
            first_idx = int(batch["sense_ids"][i][0])
            last_idx = int(batch["sense_ids"][i][-1] + 1)
            select_word_embs = embed_text[i, first_idx:last_idx, :]
            word_emb = select_word_embs.mean(dim=0)
            encoder_output_list.append(word_emb)
        encoder_output = torch.stack(encoder_output_list, dim=0) # (batch, 768)
        
        encoder_output_norm = self.batch_norm(encoder_output)
        hidden_output = self.dropout(self.act_fun(self.hidden_MLP(encoder_output_norm)))
        
        return self.classifier(hidden_output)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, eps=self.hparams.adam_eps, weight_decay=self.hparams.wd)
        reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min', verbose=True, min_lr=self.hparams.min_lr, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr_on_plateau,
                "monitor": 'val_loss',
                "frequency": 1
            },
        }
        
    def coarse_loss_oriented(self, outputs, labels, candidates):
        add_loss = 0
        for i in range(len(outputs)):
            fine_candidates_pred = torch.index_select(outputs[i], 0, torch.tensor(candidates[i]).to(self.device))
            fine_best_prediction = torch.argmax(fine_candidates_pred, dim=0)
            coarse_best_prediction = self.fine2coarse[ self.fine_id2sense[candidates[i][fine_best_prediction.item()]] ]
            
            if coarse_best_prediction == self.fine2coarse[ self.fine_id2sense[labels[i]] ]:
                add_loss += 1
        return - add_loss/len(outputs)

    def loss_function(self, outputs, labels, candidates):
        assert self.hparams.coarse_loss_oriented == False or self.hparams.coarse_or_fine == "fine"
        cross_entropy_loss = nn.CrossEntropyLoss()
        labels = torch.tensor(labels).to(self.device)
        loss = cross_entropy_loss(outputs, labels)
        # I create an aditional loss which "rewards" the fine-grained model if predicts the correct homonym set
        if self.hparams.coarse_loss_oriented:
            loss += self.coarse_loss_oriented(outputs, labels, candidates)
        return {"loss": loss}
    
    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self(batch)
        loss = self.loss_function(outputs, labels, batch["candidates"])
        self.log_dict(loss)
        # since we only monitor the loss for the training phase, we don't need to write additional 
        # code in the 'training_epoch_end' function!
        return {'loss': loss['loss']}

    def predict(self, batch):
        assert self.hparams.predict_coarse_with_fine == False or self.hparams.coarse_or_fine == "fine"
        assert self.hparams.predict_fine_with_coarse_filter == False or self.hparams.coarse_or_fine == "fine"
        with torch.no_grad():
            candidates = batch["candidates"]
            outputs = self(batch)
            ris = []
            if self.hparams.predict_fine_with_coarse_filter:
                coarse_filter_preds = self.coarse_filter_model.predict(batch)
                for i in range(len(outputs)):
                    if 2158 in candidates[i]: # if in the candidates set there is at least one <UNK> sense, we directly predict <UNK> !
                        ris.append(2158)
                        continue
                    fine_senses = list((self.coarse2fine[coarse_filter_preds[i]]).keys())
                    fine_senses_ids = [self.fine_sense2id[e] for e in fine_senses]
                    candidates_pred = torch.index_select(outputs[i], 0, torch.tensor(fine_senses_ids).to(self.device))
                    best_prediction = torch.argmax(candidates_pred, dim=0)
                    ris.append(candidates[i][best_prediction.item()])
            else:
                for i in range(len(outputs)):
                    if 2158 in candidates[i]: # if in the candidates set there is at least one <UNK> sense, we directly predict <UNK> !
                        ris.append(2158)
                        continue
                    candidates_pred = torch.index_select(outputs[i], 0, torch.tensor(candidates[i]).to(self.device))
                    best_prediction = torch.argmax(candidates_pred, dim=0)
                    if best_prediction == 0 and len(candidates[i]) != 1:
                        self.first_sense_statistic += 1
                    if self.hparams.predict_coarse_with_fine == True:
                        ris.append(self.fine2coarse[ self.fine_id2sense[candidates[i][best_prediction.item()]] ])
                    else:
                        ris.append(candidates[i][best_prediction.item()])
            return ris # list of predicted senses (expressed in indices)
 
    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self(batch)
        # LOSS
        val_loss = self.loss_function(outputs, labels, batch["candidates"])["loss"]
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        # F1-SCORE
        # good practice to follow with pytorch_lightning for logging values each iteration!
  		# https://github.com/Lightning-AI/lightning/issues/4396
        preds = self.predict(batch)
        self.val_micro_f1.update(torch.tensor(preds), torch.tensor(labels))
        self.log("val_micro_f1", self.val_micro_f1, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)    
  
      
class WSD_Gloss_Model(pl.LightningModule):
    def __init__(self, hparams):
        super(WSD_Gloss_Model, self).__init__()
        self.save_hyperparameters(hparams)
        if self.hparams.encoder_type == "bert":
            self.encoder = BertModel.from_pretrained("bert-base-uncased")
        elif self.hparams.encoder_type == "roberta":
            self.encoder = RobertaModel.from_pretrained("roberta-base")
        elif self.hparams.encoder_type == "deberta":
            self.encoder = DebertaModel.from_pretrained("microsoft/deberta-base")
        
        # we set all parameters to be not trainable
        for param in self.encoder.parameters():
            param.requires_grad = False
        # here we decide which parameters unfreeze
        if self.hparams.fine_tune_bert is True:
            if self.hparams.encoder_type == "bert":
                unfreeze = [10,11]
            elif self.hparams.encoder_type == "roberta":
                unfreeze = [10,11]
            elif self.hparams.encoder_type == "deberta":
                unfreeze = [10,11]
            for i in unfreeze:
                for param in self.encoder.encoder.layer[i].parameters():
                    param.requires_grad = True
        
        self.batch_norm = nn.BatchNorm1d(768)
        self.hidden_MLP = nn.Linear(768, self.hparams.hidden_dim, bias=True)
        if self.hparams.act_fun == "relu":
            self.act_fun = nn.ReLU(inplace=True)
        if self.hparams.act_fun == "silu":
            self.act_fun = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.classifier = nn.Linear(self.hparams.hidden_dim, 1, bias=False) # final linear projection with no bias (BINARY CLASSIFICATION)
        
        self.val_binary_micro_f1 = F1Score(task="binary", num_classes=2, average="micro")
       
    def forward(self, batch):
        text = batch["inputs"]
        if self.hparams.encoder_type == "roberta": # roberta doesn't need "token_type_ids"
            embed_text = self.encoder(text["input_ids"], attention_mask=text["attention_mask"], output_hidden_states=True)
        else:
            embed_text = self.encoder(text["input_ids"], attention_mask=text["attention_mask"], token_type_ids=text["token_type_ids"], output_hidden_states=True)
        # I take the hidden representation of the last four layers of each token
        if self.hparams.sum_or_mean == "sum":
            embed_text = torch.stack(embed_text.hidden_states[-4:], dim=0).sum(dim=0)
        elif self.hparams.sum_or_mean == "mean":
            embed_text = torch.stack(embed_text.hidden_states[-4:], dim=0).mean(dim=0)
        
        # I select the embeddings of the word we want to disambiguate and take their average! (for each item in the batch)
        encoder_output_list = []
        for i in range(len(batch["sense_ids"])):
            first_idx = int(batch["sense_ids"][i][0])
            last_idx = int(batch["sense_ids"][i][-1] + 1)
            select_word_embs = embed_text[i, first_idx:last_idx, :]
            word_emb = select_word_embs.mean(dim=0)
            encoder_output_list.append(word_emb)
        encoder_output = torch.stack(encoder_output_list, dim=0) # (batch, 768)
        
        encoder_output_norm = self.batch_norm(encoder_output)
        hidden_output = self.dropout(self.act_fun(self.hidden_MLP(encoder_output_norm)))
        output = self.classifier(hidden_output)
        
        return output.squeeze(1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, eps=self.hparams.adam_eps, weight_decay=self.hparams.wd)
        reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min', verbose=True, min_lr=self.hparams.min_lr, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr_on_plateau,
                "monitor": 'val_loss',
                "frequency": 1
            },
        }

    def loss_function(self, outputs, labels):
        binary_cross_entropy_loss = nn.BCEWithLogitsLoss()
        labels = torch.tensor(labels).to(self.device).float()
        loss = binary_cross_entropy_loss(outputs, labels)
        return {"loss": loss}
    
    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self(batch)
        loss = self.loss_function(outputs, labels)
        self.log_dict(loss)
        # since we only monitor the loss for the training phase, we don't need to write additional 
        # code in the 'training_epoch_end' function!
        return {'loss': loss['loss']}

    def predict(self, batch):
        with torch.no_grad():
            outputs = self(batch)
            ris = []
            for i in range(len(outputs)):
                prediction = torch.argmax(outputs[i], dim=0)
                ris.append(int(prediction.item()))
            return ris
 
    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self(batch)
        # LOSS
        val_loss = self.loss_function(outputs, labels)["loss"]
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        # F1-SCORE
        # good practice to follow with pytorch_lightning for logging values each iteration!
  		# https://github.com/Lightning-AI/lightning/issues/4396
        preds = self.predict(batch)
        self.val_binary_micro_f1.update(torch.tensor(preds), torch.tensor(labels))
        self.log("val_binary_micro_f1", self.val_binary_micro_f1, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)