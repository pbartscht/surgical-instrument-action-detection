#!/usr/bin/env python3
"""
Erweitertes YOLO Training Script mit ConfMix
==========================================
Unterstützt:
- CholecT50 Original Daten
- Mixed Samples mit spezifischer Label-Struktur
Debug-Version für Mixed Samples Überprüfung
"""

import os
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
from pathlib import Path
import wandb
from torch.utils.data import Dataset, DataLoader
from ultralytics.data.dataset import YOLODataset
import yaml 
import torch.nn.functional as F

# Pfade
YOLO_MODEL_PATH = "/data/Bartscht/YOLO/best_v35.pt"
CHOLECT50_PATH = "/data/Bartscht/YOLO"
MIXED_SAMPLES_PATH = "/data/Bartscht/mixed_samples"
DATA_YAML_PATH = "/home/Bartscht/YOLO/surgical-instrument-action-detection/models/hierarchical-surgical-workflow/Instrument-classification-detection/config/model_config/data.yaml"
ADAPTATION_OUTPUT_DIR = "/data/Bartscht/domain_adaptation"
NUM_EPOCHS = 100  
SAVE_INTERVAL = 10  

class MixedSamplesDataset(Dataset):
    """Dataset-Klasse für die Mixed Samples mit Debug-Ausgaben"""
    def __init__(self, mixed_samples_path):
        self.mixed_samples_path = Path(mixed_samples_path)
        self.images_path = self.mixed_samples_path / "images"
        self.labels_path = self.mixed_samples_path / "labels"
        
        # Debug: Überprüfe Verzeichnisse
        print(f"\nMixedSamplesDataset Initialisierung:")
        print(f"- Images Pfad existiert: {self.images_path.exists()}")
        print(f"- Labels Pfad existiert: {self.labels_path.exists()}")
        
        # Alle mixed sample Bilder finden
        self.image_files = list(self.images_path.glob("*.png"))
        print(f"- Gefundene Mixed Samples: {len(self.image_files)}")
        print(f"- Erster Bildpfad: {self.image_files[0] if self.image_files else 'Keine Bilder gefunden'}")
        
        # Debug: Überprüfe Labels
        if self.image_files:
            first_label = self.labels_path / f"{self.image_files[0].stem}.txt"
            if first_label.exists():
                print(f"- Beispiel Label-Datei existiert: {first_label}")
                print("- Ersten paar Zeilen des Labels:")
                with open(first_label, 'r') as f:
                    print(f.readlines()[:5])
            else:
                print("WARNUNG: Keine Label-Datei gefunden für erstes Bild!")
        else:
            print("WARNUNG: Keine Mixed Sample Bilder gefunden!")

    def parse_mixed_sample_label(self, label_file):
        """Parsed die spezielle Label-Datei der Mixed Samples"""
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # Initialisiere Datencontainer
        labels = {
            'source_image': None,
            'target_region': None,
            'confidence': None,
            'source_labels': [],
            'target_detections': []
        }

        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('Source:'):
                labels['source_image'] = line.split('Source:')[1].strip()
            elif line.startswith('Target region:'):
                region_str = line.split('Target region:')[1].strip()
                labels['target_region'] = eval(region_str)
            elif line.startswith('Confidence:'):
                labels['confidence'] = float(line.split('Confidence:')[1].strip())
            elif line == 'Source labels:':
                current_section = 'source'
            elif line == 'Target detections:':
                current_section = 'target'
            else:
                if current_section == 'source':
                    labels['source_labels'].append(line)
                elif current_section == 'target':
                    labels['target_detections'].append(line)

        return labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Lade Bild
        image_file = self.image_files[idx]
        image = Image.open(image_file)
        
        # Lade zugehörige Labels
        label_file = self.labels_path / f"{image_file.stem}.txt"
        labels = self.parse_mixed_sample_label(label_file)
        
        # Debug: Überprüfe Bildgröße und Label-Format
        if idx == 0:  # Nur beim ersten Aufruf
            print(f"\nErster Datensatz-Zugriff:")
            print(f"- Bildgröße: {image.size}")
            print(f"- Label Struktur: {labels.keys()}")
        
        # Konvertiere Bild zu Tensor
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
        
        return {
            'image': image_tensor,
            'labels': labels,
            'image_file': str(image_file)
        }

class ConfMixTrainer:
    def __init__(self, pretrained_model_path, CHOLECT50_PATH, mixed_samples_path):
        # Initialize model and datasets
        self.model = YOLO(pretrained_model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize datasets
        data_config = os.path.join(CHOLECT50_PATH, 'data.yaml')
        
        self.mixed_dataset = MixedSamplesDataset(mixed_samples_path)
        
        # DataLoaders
        self.cholect50_loader = DataLoader(
            self.cholect50_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4
        )
        
        self.mixed_loader = DataLoader(
            self.mixed_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4
        )
        
        # Progressive confidence parameters
        self.alpha = 5  # Controls transition speed
        
    def compute_confidence_transition(self, iteration, total_iterations):
        """Progressive confidence transition from Cdet to Ccomb"""
        r = iteration / total_iterations
        delta = (2 / (1 + np.exp(-self.alpha * r))) - 1
        return delta
        
    def compute_consistency_loss(self, mixed_pred, target_pred, source_pred, mixing_mask):
        """Compute consistency loss between mixed predictions and source/target"""
        loss = 0
        
        # Apply mixing mask to predictions
        mixed_masked = mixed_pred * mixing_mask
        source_masked = source_pred * (1 - mixing_mask)
        target_masked = target_pred * mixing_mask
        
        # Combined source and target predictions
        combined_pred = source_masked + target_masked
        
        # MSE loss between mixed and combined predictions
        loss = F.mse_loss(mixed_masked, combined_pred)
        
        return loss
        
    def compute_detection_loss(self, predictions, targets):
        """Standard YOLO detection loss on source domain"""
        return self.model.compute_loss(predictions, targets)
    
    def train_epoch(self, epoch, total_epochs):
        epoch_loss = 0
        num_batches = min(len(self.cholect50_loader), len(self.mixed_loader))
        
        for batch_idx, (cholect50_batch, mixed_batch) in enumerate(
            zip(self.cholect50_loader, self.mixed_loader)):
            
            # Get current confidence transition weight
            current_iteration = epoch * num_batches + batch_idx
            total_iterations = total_epochs * num_batches
            delta = self.compute_confidence_transition(current_iteration, total_iterations)
            
            # 1. Standard detection loss on source domain (CholecT50)
            source_pred = self.model(cholect50_batch['image'].to(self.device))
            detection_loss = self.compute_detection_loss(source_pred, cholect50_batch['labels'])
            
            # 2. Consistency loss on mixed samples
            mixed_images = mixed_batch['image'].to(self.device)
            mixing_mask = mixed_batch['mixing_mask'].to(self.device)
            
            # Get predictions
            mixed_pred = self.model(mixed_images)
            target_pred = self.model(mixed_batch['target_image'].to(self.device))
            source_pred = self.model(mixed_batch['source_image'].to(self.device))
            
            consistency_loss = self.compute_consistency_loss(
                mixed_pred, target_pred, source_pred, mixing_mask)
            
            # Calculate gamma based on confident predictions
            gamma = len([p for p in mixed_pred if p.conf > 0.5]) / len(mixed_pred)
            
            # Combined loss with progressive weighting
            total_loss = detection_loss + gamma * consistency_loss
            
            # Optimization step
            total_loss.backward()
            self.model.optimizer.step()
            self.model.optimizer.zero_grad()
            
            epoch_loss += total_loss.item()
            
            # Logging
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}/{total_epochs}, Batch {batch_idx}/{num_batches}")
                print(f"Detection Loss: {detection_loss.item():.4f}")
                print(f"Consistency Loss: {consistency_loss.item():.4f}")
                print(f"Delta (Confidence Transition): {delta:.4f}")
                print(f"Gamma (Consistency Weight): {gamma:.4f}")
        
        return epoch_loss / num_batches
    
    def train(self, num_epochs=50):
        """Main training loop"""
        print("Starting ConfMix training...")
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = self.train_epoch(epoch, num_epochs)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Average Loss: {epoch_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.model.save(f'confmix_checkpoint_epoch_{epoch+1}.pt')
        
        # Save final model
        self.model.save('confmix_final.pt')
        print("Training completed!")

def main():
    """
    Hauptfunktion für Unsupervised Domain Adaptation
    """
    # Erstelle Ausgabeverzeichnis
    os.makedirs(ADAPTATION_OUTPUT_DIR, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="surgical-instrument-detection",
        name="confmix_training"
    )
    
    # Initialize trainer
    trainer = ConfMixTrainer(
        pretrained_model_path=YOLO_MODEL_PATH,
        CHOLECT50_PATH=CHOLECT50_PATH,
        mixed_samples_path=MIXED_SAMPLES_PATH
    )
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()