import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def train_model(data, model, experiment_name, patience, metric_to_monitor, mode, epochs, precision):
    logger =  WandbLogger()
    logger.experiment.watch(model, log = None, log_freq = 100000)
    early_stop_callback = EarlyStopping(
        monitor=metric_to_monitor, mode=mode, min_delta=0.00, patience=patience, verbose=True)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor=metric_to_monitor, mode=mode, dirpath="../../model/checkpoints",
        filename=experiment_name + "-{epoch:02d}-{val_micro_f1:.4f}", verbose=True)
    
    n_gpus = 1 if torch.cuda.is_available() else 0
    trainer = Trainer(
        logger=logger, max_epochs=epochs, log_every_n_steps=1, gpus=n_gpus,
        callbacks=[early_stop_callback, checkpoint_callback],
        num_sanity_val_steps=0, precision=precision
        )
    trainer.fit(model, data)
