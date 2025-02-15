import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

from confmix_core import ConfidenceBasedDetector, ConfMixDetector
from dataset_loader import create_dataloaders
from confmix_trainer import ConfMixTrainer

def main():
    # Pfade
    base_path = Path("/home/Bartscht/YOLO/surgical-instrument-action-detection")
    inference_weights = base_path / "models/hierarchical-surgical-workflow/Instrument-classification-detection/weights/instrument_detector/best_v35.pt"
    source_path = Path("/data/Bartscht/YOLO")
    target_path = Path("/data/Bartscht/HeiChole/domain_adaptation/train")
    
    # Output directory für die Ergebnisse
    save_dir = base_path / "runs/train/confmix_adaptation"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model laden
    print(f'Loading pre-trained model from {inference_weights}...')
    model = YOLO(inference_weights)
    
    # ConfMix Komponenten initialisieren
    confidence_detector = ConfidenceBasedDetector(model)
    confmix_detector = ConfMixDetector(confidence_detector)

    # Dataloaders erstellen
    print('Creating dataloaders...')
    source_loader, target_loader = create_dataloaders(
        source_path=source_path,
        target_path=target_path,
        batch_size=8
    )

    # Trainer initialisieren
    trainer = ConfMixTrainer(
        model=model,
        confmix_detector=confmix_detector,
        device=device,
        batch_size=8,
        epochs=50,
        save_dir=save_dir
    )

    # Training starten
    print('Starting ConfMix training...')
    try:
        trainer.train(source_loader, target_loader)
        print(f"Training completed. Models saved to {save_dir}")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()