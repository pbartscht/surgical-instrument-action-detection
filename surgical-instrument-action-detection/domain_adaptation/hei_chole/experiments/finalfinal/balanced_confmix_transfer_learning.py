from pathlib import Path
from ultralytics import YOLO
import yaml
import os

# Set CUDA memory allocation to be more efficient
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def load_dataset_config(yaml_path):
    """
    Lädt und validiert die bestehende dataset.yaml
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded dataset config from {yaml_path}")
    print(f"Number of classes: {config.get('nc')}")
    print(f"Class weights: {config.get('class_weights', 'No weights found')}")
    return config

def setup_transfer_learning(pretrained_model_path, dataset_yaml_path, project_name="epoch0"):
    """
    Konfiguriert und startet das Transfer Learning mit speicheroptimierten Parametern
    """
    # Lade und validiere Dataset-Konfiguration
    dataset_config = load_dataset_config(dataset_yaml_path)
    
    # Vortrainiertes Modell laden
    model = YOLO(pretrained_model_path)
    
    # Speicheroptimierte Trainingsparameter
    training_args = {
        # Dataset Parameter
        'data': str(dataset_yaml_path),
        'imgsz': 640,             
        
        # Training Dauer und Batch-Größe
        'epochs': 35,
        'batch': 16,                
        'patience': 10,
        
        # Learning Rate Parameter
        'lr0': 0.001,
        'lrf': 0.01,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        
        # Optimizer Einstellungen
        'optimizer': 'AdamW',
        'weight_decay': 0.001,
        'momentum': 0.937,
        
        # Speicheroptimierung
        'cache': False,            # Kein Cache
        'workers': 4,              # Weniger Worker
        'overlap_mask': False,     # Reduziere Mask-Komplexität
        
        # Regularisierung (reduziert)
        'dropout': 0.1,
        'label_smoothing': 0.1,
        'mixup': 0.0,             # Mixup deaktiviert
        'copy_paste': 0.0,        # Copy-Paste deaktiviert
        
        # Hardware und Performance
        'device': 0,
        'amp': True,              # Automatic Mixed Precision
        'multi_scale': False,     # Multi-scale training deaktiviert
        
        # Projekt Einstellungen
        'project': project_name,
        'name': 'transfer_learning',
        'exist_ok': True,
        'pretrained': True,
        'save_period': 5,
        
        # Validierung
        'val': True,
        'save': True,
    }
    
    # Training starten
    try:
        results = model.train(**training_args)
        return results
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def main():
    # Pfade definieren
    PRETRAINED_MODEL = "/data/Bartscht/YOLO/best_v35.pt"
    DATASET_YAML = "/data/Bartscht/balanced_mixed_samples_epoch0/dataset.yaml"
    
    print("Starting transfer learning with memory-optimized settings...")
    print(f"Using pretrained model: {PRETRAINED_MODEL}")
    print(f"Using dataset config: {DATASET_YAML}")
    
    # Transfer Learning starten
    try:
        results = setup_transfer_learning(
            PRETRAINED_MODEL,
            DATASET_YAML,
            project_name="heichole_transfer_balanced_instruments"
        )
        
        print("\nTraining completed successfully!")
        print("Results summary:")
        print(f"Best mAP: {results.maps}")
        print(f"Final epoch: {results.epoch}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")

if __name__ == "__main__":
    main()