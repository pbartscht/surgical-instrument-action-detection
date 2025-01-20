import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from PIL import Image
import json
import os
from pathlib import Path
import torchvision.transforms as transforms
import numpy as np

# Mapping dictionaries remain the same
CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING = {
    'grasper': 'grasper',
    'bipolar': 'coagulation',
    'clipper': 'clipper',
    'hook': 'coagulation',
    'scissors': 'scissors',
    'irrigator': 'suction_irrigation'
}

TOOL_MAPPING = {
    0: 'grasper', 1: 'bipolar', 2: 'hook', 
    3: 'scissors', 4: 'clipper', 5: 'irrigator'
}

class HeiCholeDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((640, 640)),  # YOLO default size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.samples = []
        self._load_dataset()
    
    def _load_dataset(self):
        videos_dir = self.dataset_dir / "Videos"
        labels_dir = self.dataset_dir / "Labels"
        
        for video_dir in videos_dir.glob("*"):
            if not video_dir.is_dir():
                continue
                
            video_name = video_dir.name
            json_file = labels_dir / f"{video_name}.json"
            
            if not json_file.exists():
                continue
                
            # Load annotations
            with open(json_file, 'r') as f:
                annotations = json.load(f)
            
            # Process each frame
            for frame_file in video_dir.glob("*.png"):
                frame_number = int(frame_file.stem)
                frame_data = annotations['frames'].get(str(frame_number), {})
                
                # Get binary instrument labels
                instruments = frame_data.get('instruments', {})
                labels = torch.zeros(len(CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING))
                
                for instr_name, present in instruments.items():
                    if present > 0:
                        # Map HeiChole instrument back to CholecT50 format
                        for idx, cholect_instr in TOOL_MAPPING.items():
                            if CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING.get(cholect_instr) == instr_name:
                                labels[idx] = 1
                                break
                
                self.samples.append({
                    'image_path': str(frame_file),
                    'labels': labels,
                    'video': video_name,
                    'frame': frame_number
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path'])
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'labels': sample['labels'],
            'video': sample['video'],
            'frame': sample['frame']
        }



class FeatureAlignmentHead(nn.Module):
    """
    Neural network head for feature alignment and binary classification.
    Uses a shared feature space for both domain adaptation and classification tasks.
    """
    def __init__(self, num_classes):
        super().__init__()
        # Shared feature extractor
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(512, 256, 1),  # Reduce channel dimension
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten()
        )
        
        # Domain classifier with gradient reversal for adversarial training
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Binary instrument classifier
        self.instrument_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x, alpha=1.0):
        features = self.feature_reducer(x)
        
        # Gradient reversal scaling for domain adaptation
        domain_pred = self.domain_classifier(
            GradientReversal.apply(features, alpha)
        )
        class_pred = self.instrument_classifier(features)
        
        return domain_pred, class_pred

class GradientReversal(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial domain adaptation.
    Forward pass is identity function, backward pass reverses gradients.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DomainAdaptationTrainer:
    """
    Trainer class for domain adaptation using feature alignment.
    Only requires binary classification labels from HeiChole dataset.
    """
    def __init__(self, yolo_path, device, save_dir):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and freeze YOLO model
        self.yolo_model = YOLO(yolo_path)
        self.yolo_model.model.eval()
        
        # Initialize feature alignment head
        self.alignment_head = FeatureAlignmentHead(
            num_classes=len(TOOL_MAPPING)
        ).to(device)
        
        # Optimizer for alignment head only
        self.optimizer = torch.optim.Adam(
            self.alignment_head.parameters(),
            lr=0.001,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=5,
            factor=0.5
        )
    
    def extract_features(self, images):
        """Extract features from YOLO's C2PSA layer"""
        x = images
        features = None
        
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model.model.model):
                x = layer(x)
                if i == 10:  # C2PSA layer
                    features = x
                    break
        
        return features
    
    def train_step(self, batch, epoch_progress):
        """
        Single training step with feature alignment and classification.
        Uses gradient reversal for domain adaptation.
        """
        self.alignment_head.train()
        self.optimizer.zero_grad()
        
        # Get images and labels
        images = batch['image'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Extract features using frozen YOLO
        features = self.extract_features(images)
        
        # Calculate adaptation factor (gradually increase)
        p = epoch_progress
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        # Forward pass through alignment head
        domain_pred, class_pred = self.alignment_head(features, alpha)
        
        # Classification loss (binary cross-entropy)
        clf_loss = F.binary_cross_entropy(
            class_pred,
            labels,
            reduction='mean'
        )
        
        # Domain adaptation loss
        domain_target = torch.ones_like(domain_pred)  # Target domain = 1
        domain_loss = F.binary_cross_entropy(
            domain_pred,
            domain_target,
            reduction='mean'
        )
        
        # Combined loss
        total_loss = clf_loss + 0.1 * domain_loss
        
        # Backward pass and optimization
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'clf_loss': clf_loss.item(),
            'domain_loss': domain_loss.item()
        }
    
    def validate(self, val_loader):
        """Validate model performance"""
        self.alignment_head.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                features = self.extract_features(images)
                _, predictions = self.alignment_head(features)
                
                loss = F.binary_cross_entropy(predictions, labels)
                total_loss += loss.item()
                
                # Calculate accuracy
                pred_classes = (predictions > 0.5).float()
                correct += (pred_classes == labels).float().sum()
                total += labels.numel()
        
        return {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': (correct / total).item()
        }
    
    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.alignment_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        save_path = self.save_dir / f'alignment_head_epoch_{epoch}.pt'
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")

def train_domain_adaptation(args):
    """Main training loop for domain adaptation"""
    trainer = DomainAdaptationTrainer(
        yolo_path=args.yolo_path,
        device=args.device,
        save_dir=args.save_dir
    )
    
    # Setup data loaders
    train_dataset = HeiCholeDataset(
        dataset_dir=args.data_dir,
        transform=transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                              [0.229, 0.224, 0.225])
        ])
    )
    
    # Split into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        epoch_losses = []
        
        # Training phase
        for i, batch in enumerate(train_loader):
            progress = (epoch + i / len(train_loader)) / args.num_epochs
            losses = trainer.train_step(batch, progress)
            epoch_losses.append(losses)
        
        # Calculate average training metrics
        avg_losses = {
            k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
            for k in epoch_losses[0].keys()
        }
        
        # Validation phase
        val_metrics = trainer.validate(val_loader)
        
        # Update learning rate
        trainer.scheduler.step(val_metrics['val_loss'])
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            trainer.save_checkpoint(epoch, {**avg_losses, **val_metrics})
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"Train Loss: {avg_losses['total_loss']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['val_accuracy']:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_path', type=str, required=True,
                      help='Path to pretrained YOLO model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to HeiChole dataset')
    parser.add_argument('--save_dir', type=str, required=True,
                      help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    train_domain_adaptation(args)