from dataclasses import dataclass

@dataclass
class Hparams:
    
    ## dataloader params
    prefix_path: str = ""
    data_train: str = "data/train.jsonl"
    data_val: str = "data/dev.jsonl"
    data_test: str = "data/test.jsonl"
    vocab_path: str = "model/files/vocabs/word2id.json"
    window_size: int = 40
    window_shift: int = 40
    batch_size: int = 128 # size of the batches
    n_cpu: int = 8  # number of cpu threads to use for the dataloaders
    pin_memory: bool = False # parameter to pin memory in dataloader
    
    ## train params
    lr: float = 2e-4 # 2e-4 or 1e-3
    min_lr: float = 1e-8 # min lr for ReduceLROnPlateau
    adam_eps: float = 1e-6 # term added to the denominator to improve numerical stability
    wd: float = 1e-6 # weight decay as regulation strategy
    log_each_step: int = 20
    load_pretrained_emb: bool = True
    finetune_emb: bool = True
    # embedding strategy
    num_emb: int = 2
    stop_train_emb: int = 30
    # mixing windows strategy
    change_window_each_epoch: int = -1 #2
    
    ## extras
    POS_emb: bool = False
    positional_encode: bool = False
    
    ## model params
    # embedding
    emb_folder: str = "model/embeddings/"
    vocab_size: int = 400009
    emb_dim: int = 300
    # lstm
    hidden_dim: int = 440
    bidirectional: bool = True
    num_layers: int = 5
    dropout: float = 0.4
    # classifier
    mlp: bool = False
    num_classes: int = 11