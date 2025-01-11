#!/usr/bin/env python3
"""
Training Script for CholecT50 Instrument-Verb Detection Model

This script implements the training pipeline for a deep learning model
designed to detect instrument-verb pairs in surgical videos from the CholecT50 dataset.
"""

import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.data_iv import CholecT50_DataModule
from models.cholect50_multitask_model import CholecT50Model

def main():
    # Basic setup
    base_dir = "/data/Bartscht/CholecT50"
    batch_size = 64
    
    # Initialize data and model
    print("Initializing DataModule...")
    datamodule = CholecT50_DataModule(base_dir, batch_size)
    
    print("Initializing Model...")
    model = CholecT50Model()
    
    # Initialize WandB logger first to get the run ID
    wandb_logger = WandbLogger(project='iv-resnet50-project')
    run_name = wandb_logger.experiment.name
    
    # Create a unique directory for this run's checkpoints
    checkpoint_dir = os.path.join('checkpoints', f'{run_name}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Updated checkpoint callback with run information
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{run_name}-epoch-' + '{epoch:02d}',
        monitor='val/total_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    # Simple trainer setup
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0, 
        gradient_clip_algorithm="norm",
        deterministic=True
    )
    
    # Start training
    try:
        print(f"Starting training... Run Name: {run_name}")
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main()