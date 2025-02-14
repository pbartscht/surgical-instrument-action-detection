import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import random
import numpy as np
from torchvision import transforms

class ConfMixDataset(Dataset):
    def __init__(self, source_path, target_path, confmix_detector, image_size=(640, 640)):
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.confmix_detector = confmix_detector
        self.image_size = image_size
        
        # Load image paths
        self.source_images = list((self.source_path / "images" / "train").glob("*.png"))
        self.source_labels = self.source_path / "labels" / "train"
        self.target_images = list(self.target_path.rglob("*.png"))
        
        print(f"Loaded {len(self.source_images)} source images")
        print(f"Loaded {len(self.target_images)} target images")
        
        # Base transforms for resizing
        self.resize_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        
        # Augmentation transforms
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2))
        ])

    def __len__(self):
        return len(self.target_images)
    
    def _load_and_resize_image(self, image_path):
        """Load and resize image while preserving aspect ratio"""
        image = Image.open(image_path).convert('RGB')
        return self.resize_transform(image)

    def _load_labels(self, label_path):
        """Load and parse YOLO format labels"""
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        labels.append({
                            'class': int(parts[0]),
                            'box': [float(x) for x in parts[1:]]
                        })
        return labels

    def __getitem__(self, idx):
        # Calculate progress ratio
        progress_ratio = idx / len(self)
        
        # Load and resize target image
        target_path = self.target_images[idx]
        target_tensor = self._load_and_resize_image(target_path)
        
        # Get random source image and labels
        source_path = random.choice(self.source_images)
        source_tensor = self._load_and_resize_image(source_path)
        source_labels = self._load_labels(self.source_labels / f"{source_path.stem}.txt")
        
        # Process with ConfMix detector
        target_image = transforms.ToPILImage()(target_tensor)
        source_image = transforms.ToPILImage()(source_tensor)
        
        target_results = self.confmix_detector.process_frame(target_image, progress_ratio)
        mixed_data = self.confmix_detector.create_mixed_sample(
            target_image,
            source_image,
            target_results
        )
        
        # Convert mixed image to tensor and apply augmentations
        mixed_tensor = self.resize_transform(mixed_data['mixed_image'])
        if random.random() > 0.5:  # 50% chance to apply augmentations
            mixed_tensor = self.augment_transform(mixed_tensor)
        
        return {
            'mixed_image': mixed_tensor,
            'source_image': source_tensor,
            'mixed_labels': mixed_data['target_detections'],
            'source_labels': source_labels,
            'mixing_mask': torch.from_numpy(mixed_data['mixing_mask']).float(),
            'progress_ratio': progress_ratio
        }

def create_confmix_dataloader(source_path, target_path, confmix_detector, 
                            batch_size=8, num_workers=0):
    dataset = ConfMixDataset(source_path, target_path, confmix_detector)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,  # Drop incomplete batches
        collate_fn=collate_confmix_batch
    )

def collate_confmix_batch(batch):
    """Custom collate function for ConfMix batches"""
    return {
        'mixed_images': torch.stack([item['mixed_image'] for item in batch]),
        'source_images': torch.stack([item['source_image'] for item in batch]),
        'mixed_labels': [item['mixed_labels'] for item in batch],
        'source_labels': [item['source_labels'] for item in batch],
        'mixing_masks': torch.stack([item['mixing_mask'] for item in batch]),
        'progress_ratio': torch.tensor([item['progress_ratio'] for item in batch])
    }