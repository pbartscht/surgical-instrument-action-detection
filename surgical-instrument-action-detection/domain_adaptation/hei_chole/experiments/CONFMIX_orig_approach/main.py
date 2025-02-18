import os
from pathlib import Path
import torch
from trainer import ConfMixTrainer

def main():
    # Set up paths
    base_path = Path("/home/Bartscht/YOLO/surgical-instrument-action-detection")
    model_path = base_path / "models/hierarchical-surgical-workflow/Instrument-classification-detection/weights/instrument_detector/best_v35.pt"
    source_path = Path("/data/Bartscht/YOLO")
    target_path = Path("/data/Bartscht/HeiChole/domain_adaptation/train")
    save_dir = base_path / "runs/train/confmix_adaptation"
    
    # Create save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    try:
        # Initialize trainer with all necessary components
        trainer = ConfMixTrainer(
            model_path=model_path,
            source_path=source_path,
            target_path=target_path,
            save_dir=save_dir,
            device=device
        )
        
        # Start training
        trainer.train()
        
    except Exception as e:
        print(f"Error during setup/training: {e}")
        raise e

if __name__ == "__main__":
    main()