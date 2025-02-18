from pathlib import Path
import yaml
import os
from ultralytics import YOLO

def load_dataset_config(yaml_path):
    """
    Loads and validates the dataset.yaml
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded dataset config from {yaml_path}")
    print(f"Number of classes: {config.get('nc')}")
    return config

def setup_training(pretrained_model_path, dataset_yaml_path, project_name="confmix_training"):
    """
    Sets up and starts the training with the mixed samples
    """
    # Load and validate dataset configuration
    dataset_config = load_dataset_config(dataset_yaml_path)
    
    # Load pretrained model
    model = YOLO(pretrained_model_path)
    
    # Training parameters optimized for mixed samples
    training_args = {
        # Dataset parameters
        'data': str(dataset_yaml_path),
        'imgsz': 640,
        
        # Training duration and batch size
        'epochs': 36,
        'batch': 16,
        'patience': 10,
        
        # Learning rate parameters
        'lr0': 0.001,  # Initial learning rate
        'lrf': 0.01,   # Final learning rate fraction
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        
        # Optimizer settings
        'optimizer': 'AdamW',  # More stable 
        'weight_decay': 0.001,
        'momentum': 0.937,
        
        # Memory optimization
        'cache': False,
        'workers': 4,
        'overlap_mask': False,
        
        # Regularization (reduced since we're using mixed samples)
        'dropout': 0.1,
        'label_smoothing': 0.1,
        'mixup': 0.0,      # Disabled 
        'copy_paste': 0.0, # Disabled 
        
        # Project settings
        'project': project_name,
        'name': 'confmix_mixed_samples',
        'exist_ok': True,
        'pretrained': True,
        'save_period': 3,
        
        # Validation
        'val': True,
        'save': True,
    }
    
    try:
        results = model.train(**training_args)
        return results
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def main():
    # Define paths
    PRETRAINED_MODEL = Path("/home/Bartscht/YOLO/surgical-instrument-action-detection/domain_adaptation/hei_chole/experiments/finalfinal/heichole_transfer_balanced_instruments/transfer_learning/weights/last.pt")
    DATASET_YAML = "/data/Bartscht/balanced_mixed_samples_epoch1/dataset.yaml"
    
    print("Starting training with ConfMix mixed samples...")
    print(f"Using pretrained model: {PRETRAINED_MODEL}")
    print(f"Using dataset config: {DATASET_YAML}")
    
    try:
        results = setup_training(
            PRETRAINED_MODEL,
            DATASET_YAML,
            project_name="confmix_balanced_1"
        )
        
        print("\nTraining completed successfully!")
        print("Results summary:")
        print(f"Best mAP: {results.maps}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")

if __name__ == "__main__":
    main()