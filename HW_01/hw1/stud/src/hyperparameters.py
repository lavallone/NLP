from dataclasses import dataclass

@dataclass
class Hparams:
    
    ## dataloader params
    prefix_path: str = ""
    data_train: str = "data/train.jsonl" # train dataset path
    data_val: str = "data/dev.jsonl" # validation dataset path
    data_test: str = "data/test.jsonl" # test dataset path
    vocab_path: str = "model/files/vocabs/word2id.json" # vocabulary path
    window_size: int = 40
    window_shift: int = 40
    batch_size: int = 512 # size of the batches
    n_cpu: int = 8  # number of cpu threads to use for the dataloaders
    pin_memory: bool = False # parameter to pin memory in dataloader
    
    ## train params
    lr: float = 1e-3 # 2e-4 or 1e-3
    min_lr: float = 1e-8 # min lr for ReduceLROnPlateau
    adam_eps: float = 1e-6 # term added to the denominator to improve numerical stability
    wd: float = 1e-6 # weight decay as regulation strategy
    log_each_step: int = 20 # logging information about the train loss each 20 training steps
    load_pretrained_emb: bool = True # load or not the pretrained word embeddings
    finetune_emb: bool = True # make thesse embeddings trainable or not
    # embedding strategy
    num_emb: int = 2 # how many embedding layers to use (in case we want to divide the GloVe embeddings with the new added ones)
    stop_train_emb: int = 20 # stop training word embeddings (if loaded) after 20 epochs
    # mixing windows strategy
    change_window_each_epoch: int = 2 # when using the 'mixing windows strategy", we change dataloader each 2 epochs
    
    ## extras
    POS_emb: bool = False # if include POS tags in the architecture or not
    positional_encode: bool = False # if adding positional encoding to the  word embeddings or not
    
    ## model params
    # embedding
    emb_folder: str = "model/embeddings/" # where the word embeddings are stored
    vocab_size: int = 400009 # size of the vocabulary
    emb_dim: int = 300 # dimension of the embeddings
    # lstm
    hidden_dim: int = 440 # hidden dimension of LSTM units
    bidirectional: bool = True # Bi-LSTMs or not
    num_layers: int = 5 # number of LSTMs stacked layers
    dropout: float = 0.5 # dropout value
    # classifier
    mlp: bool = False # use an MLP as classifier or not
    num_classes: int = 11 # number of classes to predict
    
    # this is the path of my best model to give to the StudentModel!
    student_weights_path: str = "model/checkpoints/BEST.pth" 