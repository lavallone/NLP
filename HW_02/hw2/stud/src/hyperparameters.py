from dataclasses import dataclass

@dataclass
class Hparams:
    
    ## dataloader params
    prefix_path: str = "../../"
    coarse_or_fine: str = "coarse"
    use_gloss: bool = False
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
    precision: int = 16 # 16 or 32
    use_lemmas: bool = False # use 'lemmas' as input sentences instead of 'words'
    use_POS: bool = False # use POS tags as additional information for the input sequence
    
    ## fine vs coarse
    coarse_loss_oriented: bool = False # modify loss when training a fine-grained model to positively reward correct coarse-grained predictions!
    predict_coarse_with_fine: bool = False # use a fine-grained model to predict coarse-grained senses!
    predict_fine_with_coarse_filter: bool = False # use a coarse-grained model to filter out fine-grained predictions and help the fine-grained model! 
    
    ## GLOSSES
    
    
    ## model params
    # encoder (BERT-like)
    fine_tune_bert: bool = True # make BERT layers trainable or not
    encoder_type: str = "bert" # bert, roberta or deberta
    sum_or_mean: str = "sum" # sum or mean the last four hidden states of BERT encoder
    # non-liner layer
    hidden_dim: int = 512 # I initially used 768*2 but in various papers they set 512
    act_fun: str = "silu" # 'silu' or 'relu'
    # classifier
    num_senses: int = 2158 # 2158 or 4476 number of total coarse-senses
    dropout: float = 0.6 # dropout value
    
    # this is the path of my best model to give to the StudentModel!
    student_weights_path: str = "model/checkpoints/prova-epoch=05-val_micro_f1=0.8740_lr_2e-5.ckpt"