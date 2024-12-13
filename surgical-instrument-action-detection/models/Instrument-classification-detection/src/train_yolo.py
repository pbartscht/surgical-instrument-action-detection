#!/usr/bin/env python3
"""
YOLO Training Script for Surgical Instrument Detection
===================================================

This script handles the training of a YOLO model for surgical instrument detection,
using proper relative paths for model loading and saving.
"""

import torch
import sys
from pathlib import Path
import wandb  

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Now we can import from src
from utils.custom_yolo import CustomYOLO

def main():
    # Define paths relative to project root
    pretrained_weights_path = project_root / 'weights' / 'pretrained' / 'yolo11l.pt'
    output_weights_dir = project_root / 'weights' / 'instrument_detector'
    
    # Ensure output directory exists
    output_weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Version handling for output model
    existing_versions = [int(p.stem.split('_v')[1]) for p in output_weights_dir.glob('best_v*.pt')]
    next_version = max(existing_versions, default=0) + 1
    output_model_name = f'best_v{next_version}.pt'
    
    print(f"Loading pretrained model from: {pretrained_weights_path}")
    print(f"Will save trained model to: {output_weights_dir / output_model_name}")
    
    # Initialisieren Sie wandb separat (optional)
    wandb.init(
        project="surgical-instrument-detection",
        entity="peebee-hamburg-university-of-technology",
    )
    
    # Initialize model with pretrained weights
    model = CustomYOLO(str(pretrained_weights_path))
    
    training_config = {
        # Basis-Parameter
        'data': str(project_root / 'config' / 'model_config' / 'data.yaml'),
        'epochs': 120,
        'imgsz': 512,
        'batch': 8,
        
        # Loss-Gewichtungen
        'box': 4.0,
        'cls': 3.5,
        
        # Optimierung und Regularisierung
        'lr0': 0.0005,
        'lrf': 0.01,
        'weight_decay': 0.0015,
        'warmup_epochs': 3,
        'patience': 30,
        'label_smoothing': 0.15,

        # Speicherpfad für trainiertes Modell
        'project': str(output_weights_dir.parent),
        'name': output_weights_dir.name,
        'exist_ok': True,
        
        # Rest bleibt gleich
        'amp': True,
        'cos_lr': True,
        'optimizer': 'AdamW',
        'multi_scale': True,
        'overlap_mask': True,
        'save_period': 10,
        'plots': True,
        'val': True,
        'single_cls': False,
        'rect': False
    }
    
    # Training starten
    model.train(**training_config)

if __name__ == '__main__':
    main()