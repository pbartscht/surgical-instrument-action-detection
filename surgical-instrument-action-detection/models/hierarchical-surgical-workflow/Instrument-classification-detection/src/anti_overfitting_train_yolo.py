#!/usr/bin/env python3
"""
YOLO Training Script for Surgical Instrument Detection
===================================================

Final optimized training script incorporating:
- Adjusted image size handling (640x640)
- Rectangular training for varying aspect ratios
- Optimized hyperparameters for 80 epochs
- Proper augmentations
- Excluding SpecimenBag class
"""

import torch
import sys
from pathlib import Path
import wandb
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Import custom YOLO implementation
from utils.custom_yolo import CustomYOLO

def test_dataset(model, data_yaml_path):
    """Test dataset loading and augmentations."""
    print("\n=== Testing Dataset Configuration ===")
    try:
        if not Path(data_yaml_path).exists():
            raise FileNotFoundError(f"Could not find data.yaml: {data_yaml_path}")
            
        dataset = model.get_dataset(data_yaml_path, mode='train')
        
        img, labels = dataset[0]
        print("Dataset successfully created!")
        print(f"Augmentations active: {dataset.augment}")
        print(f"Image shape: {img.shape}")
        print(f"Number of labels: {len(labels) if labels is not None else 0}")
        
    except Exception as e:
        print(f"Error testing dataset: {str(e)}")
        import traceback
        traceback.print_exc()
    print("=== Dataset test completed ===\n")

def main():
    # Define paths
    pretrained_weights_path = project_root / 'weights' / 'pretrained' / 'yolov8l.pt'
    output_weights_dir = project_root / 'weights' / 'instrument_detector'
    data_yaml_path = project_root / 'config' / 'model_config' / 'data.yaml'
    
    # Ensure output directory exists
    output_weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Version handling
    existing_versions = [int(p.stem.split('_v')[1]) for p in output_weights_dir.glob('best_v*.pt')]
    next_version = max(existing_versions, default=0) + 1
    output_model_name = f'best_v{next_version}.pt'
    
    print(f"Loading pretrained model from: {pretrained_weights_path}")
    print(f"Will save trained model to: {output_weights_dir / output_model_name}")
    
    # Initialize wandb with comprehensive config
    wandb.init(
        project="surgical-instrument-detection",
        entity="peebee-hamburg-university-of-technology",
    )
    
    # Initialize model
    model = CustomYOLO(str(pretrained_weights_path))
    
    # Test dataset configuration
    test_dataset(model, str(data_yaml_path))
    
    training_config = {
        # Base parameters
        'data': str(data_yaml_path),
        'epochs': 80,
        'imgsz': 640,  # Adjusted to match pretrained size
        'batch': 8,
        
        # Loss weights
        'box': 4.0,
        'cls': 3.5,
        
        # Optimization and regularization
        'lr0': 0.001,          # Higher initial learning rate
        'lrf': 0.05,           # Stronger LR decay
        'weight_decay': 0.004, # Increased regularization
        'warmup_epochs': 2,    # Shorter warmup
        'patience': 15,        # Earlier stopping
        'label_smoothing': 0.1,# Reduced label smoothing

        # Save configuration
        'project': str(output_weights_dir.parent),
        'name': output_weights_dir.name,
        'exist_ok': True,

        # Augmentations
        'mosaic': 1,           # Enable mosaic
        'hsv_h': 0.015,       # Mild HSV augmentation
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'augment': True,      # Enable custom augmentations
        'cache': False,
        
        # Image handling
        'rect': True,         # Enable rectangular training
        'multi_scale': True,  # Enable multi-scale training
        
        # Additional parameters
        'amp': True,          # Mixed precision training
        'cos_lr': True,       # Cosine LR scheduler
        'optimizer': 'AdamW',
        'overlap_mask': True,
        'save_period': 5,     # Save checkpoints every 5 epochs
        'plots': True,
        'val': True          # Enable validation
    }
    
    # Start training
    print("\n=== Starting Training ===")
    print("Training configuration:")
    for key, value in training_config.items():
        print(f"{key}: {value}")
    print("\n")
    
    try:
        model.train(**training_config)
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure wandb is properly closed
        wandb.finish()

if __name__ == '__main__':
    main()