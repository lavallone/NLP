import torch
import wandb
import os
import json
import random
from seqeval.metrics import f1_score
from .models import predict_function

# I took inspiration from the function "train_and_evaluate" of the NLP Notebook #4 - POS tagging
# It's however a standard Deep Learning class used to train and evaluate a model in pytorch.
class Trainer():

    def __init__(self, model, device, loss_function, optimizer, scheduler):
        self.model = model
        self.device = device
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_each_step = self.model.hparams.log_each_step
        
        # I need it for the validation phase where I compute macro f1 score
        self.val_label_true = [e["labels"] for e in [json.loads(line) for line in open(self.model.hparams.prefix_path+self.model.hparams.data_val, "r")]]
        
    def train(self, train_dataloader, train_dataloader_list, val_dataloader, val_windows_each_sentence_list, epochs, version_name, wandb_log=True, early_stopping=True, early_stopping_mode="max", early_stopping_patience=0, model_checkpoint=True):
        print('>>>>>>>>>>>>>>>> Starting Training <<<<<<<<<<<<<<<<<<<')
        print()
        
        self.model.to(self.device)
        valid_history = [(10.0, 0.0)]  # I need it for the early stopping mechanism
        patience_counter = 0 # patience for early stopping
        # for each epoch
        for epoch in range(epochs):
            
            print(f'|Epoch {epoch + 1}|')
            self.model.train()
            
            if train_dataloader_list is not None and epoch%self.model.hparams.change_window_each_epoch==0: # it means we want to use the strategy of 'mixing windows'
                print(f"* PICKED DATALOADER NUMBER {random.randint(0,len(train_dataloader_list)-1)} *")
                train_dataloader = train_dataloader_list[random.randint(0,len(train_dataloader_list)-1)]
            
            if self.model.hparams.finetune_emb and epoch == self.model.hparams.stop_train_emb: # freeze only the GloVe embeddings
                print("_________________________________________________________")
                print("______________FREEZE GLOVE EMBEDDING LAYER_______________\n")
                if self.model.hparams.num_emb == 1:
                    for param in self.model.embedding_layer.parameters():
                        param.requires_grad = False
                else:
                    for param in self.model.embedding_layer_1.parameters():
                        param.requires_grad = False

            epoch_loss = 0.0
            # for each batch 
            for step, batch in enumerate(train_dataloader):
                inputs = batch['inputs'].to(self.device)
                labels = batch['labels'].to(self.device)
                pos_inputs = None
                if self.model.hparams.POS_emb:
                    pos_inputs = batch['pos'].to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs, pos_inputs)
                
                labels = labels.view(-1) # (batch_size*window_size)
                outputs = outputs.view(-1, outputs.shape[-1]) # (batch_size*window_size, num_classes)
                batch_loss = self.loss_function(outputs, labels) # Cross Entropy Loss
                batch_loss.backward()
                self.optimizer.step()

                epoch_loss += batch_loss.tolist() # tolist() in order to have the float value of the loss

                if step % self.log_each_step == 0 and self.log_each_step != -1:
                    print(f'\t|step  {step}| --> current avg loss = {round(epoch_loss / (step + 1), 4)}')
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(f'|TRAIN| loss = {round(avg_epoch_loss, 4)}')
            print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")

            ## VAL phase
            
            val_loss, val_macro_f1 = self.validate(val_dataloader, val_windows_each_sentence_list)
            valid_history.append((val_loss, val_macro_f1))
            if wandb_log:
                wandb.log({"loss" : avg_epoch_loss})
                wandb.log({"val_loss" : val_loss})
                wandb.log({"val_macro_f1" : val_macro_f1})
            
            print(f'|VAL| val loss = {round(val_loss,4)}')
            print(f'|VAL| val macro f1 = {round(val_macro_f1,4)}')
            
            self.scheduler.step(val_loss) # ReduceOnPlateau learning rate scheduler update (I can also try to use val_macro_f1 in the 'max' mode)
            
            if early_stopping:
                stop = early_stopping_mode == 'min' and valid_history[-1][0] >= torch.min(torch.tensor([e[0] for e in valid_history[:-1]])).item()
                stop = stop or early_stopping_mode == 'max' and valid_history[-1][1] <= torch.max(torch.tensor([e[1] for e in valid_history[:-1]])).item()
                if stop:
                    if patience_counter >= early_stopping_patience:
                        print('>>>>>>>>>>>>>>>>>>  |EARLY STOP| <<<<<<<<<<<<<<<<<<<<<')
                        break
                    else:
                        print('Patience...')
                        patience_counter += 1
                else: # it means the model is improving
                    patience_counter = 0
                        
            if model_checkpoint:
                current_max = torch.max(torch.tensor([e[1] for e in valid_history[:-1]])).item()
                if valid_history[-1][1] > current_max: # I save the model into "model/checkpoints"
                    if os.path.exists(f"{self.model.hparams.prefix_path}model/checkpoints/{version_name}_f1_{round(current_max,4)}.pth"):
                        os.remove(f"{self.model.hparams.prefix_path}model/checkpoints/{version_name}_f1_{round(current_max,4)}.pth")
                    torch.save(self.model.state_dict(), f"{self.model.hparams.prefix_path}model/checkpoints/{version_name}_f1_{round(valid_history[-1][1],4)}.pth")
                
            print("_________________________________________________________")

        print('>>>>>>>>>>>>>>>>>>>>>> Done! <<<<<<<<<<<<<<<<<<<<<<<<<')
        print()
    
    def validate(self, val_dataloader, val_windows_each_sentence_list):
        self.model.eval() # very important step!
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = batch['inputs'].to(self.device)
                labels = batch['labels'].to(self.device)
                pos_inputs = None
                if self.model.hparams.POS_emb:
                    pos_inputs = batch['pos'].to(self.device)
                
                ouputs = self.model(inputs, pos_inputs)
                ouputs = ouputs.view(-1, ouputs.shape[-1])
                labels = labels.view(-1)
                batch_loss = self.loss_function(ouputs, labels)
                val_loss += batch_loss.tolist()
            
            # we use this setting for validation phase
            self.model.hparams.window_size = 30
            self.model.hparams.window_shift = 15
            val_label_pred = predict_function(self.model, self.device, self.val_label_true, val_dataloader, val_windows_each_sentence_list)
            val_macro_f1 = f1_score(self.val_label_true, val_label_pred, average="macro")
                
        return val_loss / len(val_dataloader), val_macro_f1