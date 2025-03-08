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
import numpy as np

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
    
    # Initialize wandb separately (optional)
    wandb.init(
        project="surgical-instrument-detection",
        entity="peebee-hamburg-university-of-technology",
    )
    
    # Initialize model with pretrained weights
    model = CustomYOLO(str(pretrained_weights_path))

    def test_augmentations():
        print("\n=== Testing Augmentations ===")
        try:
            data_yaml_path = str(project_root / 'config' / 'model_config' / 'dataset1.yaml')
            print(f"Loading dataset from: {data_yaml_path}")
            
            # Prüfen, ob die Datei existiert
            if not Path(data_yaml_path).exists():
                raise FileNotFoundError(f"Konnte data.yaml nicht finden: {data_yaml_path}")
                
            # Dataset erstellen mit korrekter Parameterübergabe
            dataset = model.get_dataset(data_yaml_path, mode='train')
            
            # Test des ersten Bildes
            img, labels = dataset[0]
            print("Dataset erfolgreich erstellt!")
            print(f"Augmentierungen aktiv: {dataset.augment}")
            print(f"Bildform: {img.shape}")
            print(f"Anzahl Labels: {len(labels) if labels is not None else 0}")
            
        except Exception as e:
            print(f"Fehler beim Testen der Augmentierungen: {str(e)}")
            import traceback
            traceback.print_exc()
        print("=== Augmentations-Test abgeschlossen ===\n")
    
    # Test durchführen
    test_augmentations()
    
    training_config = {
        # Basis-Parameter
        'data': str(project_root / 'config' / 'model_config' / 'dataset1.yaml'),
        'epochs': 100,
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

        'mosaic': 0,  # Deaktiviert YOLO's mosaic augmentation
        'hsv_h': 0,   # Deaktiviert YOLO's HSV augmentation
        'hsv_s': 0,
        'hsv_v': 0,
        'augment': True,  # Aktiviert die Augmentierung
        'cache': False,
        
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
    print("Initializing model...")  # Debug print
    model = CustomYOLO(str(pretrained_weights_path))
    print("Starting training...")   # Debug print
    model.train(**training_config)

if __name__ == '__main__':
    main()