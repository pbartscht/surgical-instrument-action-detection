import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

from confmix_core import ConfidenceBasedDetector, ConfMixDetector
from dataset_loader import create_dataloaders
from confmix_trainer import ConfMixTrainer

def main():
    # Paths
    base_path = Path("/home/Bartscht/YOLO/surgical-instrument-action-detection")
    inference_weights = base_path / "models/hierarchical-surgical-workflow/Instrument-classification-detection/weights/instrument_detector/best_v35.pt"
    source_path = Path("/data/Bartscht/YOLO")
    target_path = Path("/data/Bartscht/HeiChole/domain_adaptation/train")
    
    # Output directory for results
    save_dir = base_path / "runs/train/confmix_adaptation"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f'Loading pre-trained model from {inference_weights}...')
    model = YOLO(inference_weights)
    
    # Create dataloaders
    print('Creating dataloaders...')
    source_loader, target_loader = create_dataloaders(
        source_path=source_path,
        target_path=target_path,
        batch_size=8
    )

    # Initialize trainer
    # Note: ConfMixTrainer creates its own ConfidenceBasedDetector and ConfMixDetector internally
    trainer = ConfMixTrainer(
        model=model,
        device=device,
        save_dir=save_dir
    )

    # Start training loop
    print('Starting ConfMix training...')
    total_epochs = 50
    for epoch in range(total_epochs):
        progress_ratio = epoch / total_epochs
        
        # Train one epoch
        for batch_idx, (source_batch, target_batch) in enumerate(zip(source_loader, target_loader)):
            loss = trainer.train_step(source_batch, target_batch, progress_ratio)
            
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f'Epoch [{epoch}/{total_epochs}], Batch [{batch_idx}], Loss: {loss:.4f}')
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')

    print(f"Training completed. Models saved to {save_dir}")

if __name__ == "__main__":
    main()