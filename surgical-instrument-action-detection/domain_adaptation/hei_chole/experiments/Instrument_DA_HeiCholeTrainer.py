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

# Constants from evaluation code
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

class DomainAdaptationTrainer:
    def __init__(self, yolo_path, device):
        self.device = device
        
        # Load YOLO model
        self.yolo_model = YOLO(yolo_path)
        self.yolo_model.model.eval()
        
        # Initialize domain head
        self.domain_head = DomainHead(num_classes=len(TOOL_MAPPING)).to(device)
        
        # Optimizer only for domain head
        self.optimizer = torch.optim.Adam(
            self.domain_head.parameters(),
            lr=0.001
        )
        
        # Load datasets
        self.setup_datasets()
    
    def setup_datasets(self):
        # Source domain (CholecT50) loader would be set up similarly
        # For now focusing on target domain (HeiChole)
        heichole_dataset = HeiCholeDataset(
            dataset_dir="/data/Bartscht/HeiChole"
        )
        
        self.target_loader = DataLoader(
            heichole_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4
        )
    
    def extract_yolo_features(self, images):
        """Extracts features after C2PSA layer (Layer 10)"""
        x = images
        features = None
        
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model.model.model):
                x = layer(x)
                if i == 10:  # After C2PSA Layer
                    features = x
                    break
        
        return features
    
    def train_epoch(self, source_loader, target_loader):
        epoch_losses = []
        
        for source_batch, target_batch in zip(source_loader, target_loader):
            losses = self.train_step(source_batch, target_batch)
            epoch_losses.append(losses)
            
        # Calculate average losses
        avg_losses = {
            key: sum(d[key] for d in epoch_losses) / len(epoch_losses)
            for key in epoch_losses[0].keys()
        }
        
        return avg_losses
    
    def evaluate(self, loader):
        """Evaluates current model on target domain"""
        self.domain_head.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(self.device)
                features = self.extract_yolo_features(images)
                domain_out, class_out = self.domain_head(features)
                
                # Store predictions for analysis
                predictions.extend([{
                    'video': v,
                    'frame': f,
                    'predictions': p.cpu().numpy(),
                    'domain_score': d.cpu().numpy()
                } for v, f, p, d in zip(
                    batch['video'],
                    batch['frame'],
                    class_out,
                    domain_out
                )])
        
        return predictions

class DomainHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        
        # Domain classifier branch
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Instrument classifier branch
        self.task_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        shared_features = self.shared_conv(x)
        domain_output = self.domain_classifier(shared_features)
        task_output = self.task_classifier(shared_features)
        return domain_output, task_output

# Example usage
def main():
    YOLO_PATH = "/home/Bartscht/YOLO/surgical-instrument-action-detection/models/hierarchical-surgical-workflow/Instrument-classification-detection/weights/instrument_detector/best_v35.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = DomainAdaptationTrainer(YOLO_PATH, device)
    
    # Training loop would go here
    '''
    for epoch in range(num_epochs):
        losses = trainer.train_epoch(source_loader, target_loader)
        print(f"Epoch {epoch}, Losses:", losses)
        
        # Evaluate periodically
        if epoch % eval_interval == 0:
            predictions = trainer.evaluate(trainer.target_loader)
            # Analyze predictions
    '''

if __name__ == "__main__":
    main()