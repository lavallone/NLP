from dataclasses import dataclass

@dataclass
class Hparams:
    
    ## dataloader params
    prefix_path: str = "../../"
    coarse_or_fine: str = "coarse"
    data_train: str = "data/"+coarse_or_fine+"-grained/train_"+coarse_or_fine+"_grained.json" # train dataset path
    data_val: str = "data/"+coarse_or_fine+"-grained/val_"+coarse_or_fine+"_grained.json" # validation dataset path
    data_test: str = "data/"+coarse_or_fine+"-grained/test_"+coarse_or_fine+"_grained.json" # test dataset path
    sense_map: str = "data/map/coarse_fine_defs_map.json"
    batch_size: int = 64 # size of the batches
    n_cpu: int = 8 # number of cpu threads to use for the dataloaders
    pin_memory: bool = False # parameter to pin memory in dataloader
    
    ## train params
    lr: float = 2e-5 # 1e-5, 2e-5 or 1e-4
    min_lr: float = 1e-8 # min lr for ReduceLROnPlateau
    adam_eps: float = 1e-6 # term added to the denominator to improve numerical stability
    wd: float = 1e-6 # weight decay as regulation strategy
    precision: int = 16
    load_pretrained_emb: bool = True # load or not the pretrained word embeddings
    
    ## extras
    
    
    ## model params
    # encoder (BERT-like)
    fine_tune_bert: bool = True # make BERT layers trainable or not
    # classifier
    num_senses: int = 2158 # number of total coarse-senses
    dropout: float = 0.4 # dropout value
    
    # this is the path of my best model to give to the StudentModel!
    student_weights_path: str = "model/checkpoints/prova-epoch=05-val_micro_f1=0.8740_lr_2e-5.ckpt"