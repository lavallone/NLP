import torch
from torch import optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from transformers import BertModel, RobertaModel, DebertaModel
from torchmetrics import F1Score


class WSD_Model(pl.LightningModule):
    def __init__(self, hparams):
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
                unfreeze = [7,8,9,10,11]
            elif self.hparams.encoder_type == "roberta":
                unfreeze = [7,8,9,10,11]
            elif self.hparams.encoder_type == "deberta":
                unfreeze = [8,9,10,11]
            for i in unfreeze:
                for param in self.encoder.encoder.layer[i].parameters():
                    param.requires_grad = True
        
        self.batch_norm = nn.BatchNorm1d(768)
        self.hidden_MLP = nn.Linear(768, 768*2, bias=True)
        self.silu = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.classifier = nn.Linear(768*2, self.hparams.num_senses, bias=True)
        
        self.val_micro_f1 = F1Score(task="multiclass", num_classes=self.hparams.num_senses, average="micro")
       
    def forward(self, batch):
        text = batch["input"]
        embed_text = self.encoder(text["input_ids"], attention_mask=text["attention_mask"], token_type_ids=text["token_type_ids"], output_hidden_states=True)
        # I take the hidden representation of the last four layers of each token
        embed_text = torch.stack(embed_text.hidden_states[-4:], dim=0).sum(dim=0)
        
        # I select the embeddings of the word we want to disambiguate and take their average! (for each item in the batch)
        encoder_output_list = []
        for i in range(len(batch["sense_ids"])):
            first_idx = int(batch["sense_ids"][i][0])
            if self.hparams.use_POS:
                pos_idx = int(batch["sense_ids"][i][-1])
                last_idx = int(batch["sense_ids"][i][-2] + 1)
                select_word_embs = embed_text[i, first_idx:last_idx, :] + embed_text[i, pos_idx, :]
            else:
                last_idx = int(batch["sense_ids"][i][-1] + 1)
                select_word_embs = embed_text[i, first_idx:last_idx, :]
            word_emb = select_word_embs.mean(dim=0) # try also sum
            encoder_output_list.append(word_emb)
        encoder_output = torch.stack(encoder_output_list, dim=0) # (batch, 768)
        
        encoder_output_norm = self.batch_norm(encoder_output)
        hidden_output = self.dropout(self.silu(self.hidden_MLP(encoder_output_norm)))
        
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

    def loss_function(self, outputs, labels):
        loss_function = nn.CrossEntropyLoss()
        return {"loss": loss_function(outputs, torch.tensor(labels).to(self.device))}
     
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
            candidates = batch["candidates"]
            outputs = self(batch)
            ris = []
            for i in range(len(outputs)):
                if 2158 in candidates[i]: # if in the candidates set there is at least one <UNK> sense, we directly predict <UNK> !
                    ris.append(2158)
                    continue
                candidates_pred = torch.index_select(outputs[i], 0, torch.tensor(candidates[i]).to(self.device))
                best_prediction = torch.argmax(candidates_pred, dim=0)
                ris.append(candidates[i][best_prediction.item()])
            return ris # list of predicted senses (expressed in indices)
 
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
        self.val_micro_f1.update(torch.tensor(preds), torch.tensor(labels))
        self.log("val_micro_f1", self.val_micro_f1, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)