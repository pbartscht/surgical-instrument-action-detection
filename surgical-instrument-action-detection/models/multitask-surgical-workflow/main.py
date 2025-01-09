#!/usr/bin/env python3
"""
Training Script for CholecT50 Instrument-Verb Detection Model

This script implements the training pipeline for a deep learning model
designed to detect instrument-verb pairs in surgical videos from the CholecT50 dataset.

The training utilizes PyTorch Lightning for training organization and Weights & Biases
for experiment tracking.

Example:
    $ python main.py --dataset_dir /path/to/cholect50 --batch_size 64

Author: Bartscht
Date: December 2024
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.tensorboard import SummaryWriter

from utils.data_iv import CholecT50_DataModule
from models.cholect50_multitask_model import CholecT50Model

def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser.
    
    Returns:
        ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Train IV detection model on CholecT50 dataset'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default="/data/Bartscht/CholecT50",
        help='Path to CholecT50 dataset directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for training'
    )
    parser.add_argument(
        '--project_name',
        type=str,
        default="iv-resnet50-project",
        help='Project name for Weights & Biases logging'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=30,
        help='Maximum number of training epochs'
    )
    return parser

def setup_callbacks(checkpoint_dir: str, run_name: str) -> list:
    """Set up training callbacks.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        run_name: Name of the current run for checkpoint naming
        
    Returns:
        list: List of configured callbacks
    """
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{run_name}-epoch-{{epoch:02d}}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )
    
    return [checkpoint_callback, early_stopping]

def train_model(
    dataset_dir: str,
    batch_size: int,
    project_name: str,
    max_epochs: int,
    writer: Optional[SummaryWriter] = None
) -> None:
    """Train the IV detection model.
    
    Args:
        dataset_dir: Path to CholecT50 dataset
        batch_size: Training batch size
        project_name: Name for W&B project
        max_epochs: Maximum number of training epochs
        writer: Optional TensorBoard SummaryWriter
    """
    # Validate dataset directory
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
    # Initialize wandb logger first to get the run ID
    wandb_logger = WandbLogger(
        project=project_name,
        log_model=True
    )
    run_name = wandb_logger.experiment.name
    
    # Create a unique directory for this run's checkpoints
    checkpoint_dir = os.path.join('checkpoints', f'{run_name}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize data module and model
    datamodule = CholecT50_DataModule(dataset_dir, batch_size)
    model = CholecT50Model()
    
    # Configure trainer with updated callbacks
    trainer = pl.Trainer(
        accelerator="cuda",
        devices=1,
        max_epochs=max_epochs,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=setup_callbacks(checkpoint_dir, run_name),
        deterministic=True,  # For reproducibility
        gradient_clip_val=0.5,  # Prevent exploding gradients
    )
    
    try:
        print(f"Starting training... Run Name: {run_name}")
        # Train and test the model
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule)
        
        # Save final checkpoint with run name
        final_checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}-final.ckpt")
        trainer.save_checkpoint(final_checkpoint_path)
        print(f"Model saved to {final_checkpoint_path}")
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise
    
    finally:
        if writer:
            writer.close()

def main():
    """Main entry point of the training script."""
    # Parse command line arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Set up tensorboard writer
    writer = SummaryWriter("runs/RESNET50")
    
    # Configure PyTorch for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    # Start training
    try:
        train_model(
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            project_name=args.project_name,
            max_epochs=args.max_epochs,
            writer=writer
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
        
if __name__ == "__main__":
    main()