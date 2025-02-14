import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import random
import numpy as np
from base_setup import YOLOUtils
from torchvision import transforms

class ConfMixDataset(Dataset):
    def __init__(self, source_path, target_path, confmix_detector):
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.confmix_detector = confmix_detector
        
        # Load image paths
        self.source_images = list((self.source_path / "images" / "train").glob("*.png"))
        self.source_labels = self.source_path / "labels" / "train"
        self.target_images = list(self.target_path.rglob("*.png"))
        
        print(f"Loaded {len(self.source_images)} source images")
        print(f"Loaded {len(self.target_images)} target images")
        
        # YOLO augmentations
        self.transform = self._get_yolo_augmentations()
    
    def __len__(self):
        return len(self.target_images)
    
    def __getitem__(self, idx):
        # Calculate progress ratio
        progress_ratio = idx / len(self)
        
        # Load target image
        target_path = self.target_images[idx]
        target_image = Image.open(target_path)
        
        # Get random source image and labels
        source_data = self._get_random_source_data()
        
        # Process with ConfMix
        target_results = self.confmix_detector.process_frame(target_image, progress_ratio)
        mixed_data = self.confmix_detector.create_mixed_sample(
            target_image, 
            source_data['image'], 
            target_results
        )
        
        # Generate labels
        mixed_labels = self._generate_mixed_labels(
            source_data['labels'],
            target_results['detections'],
            mixed_data['selected_region'],
            target_image.size
        )
        
        # Apply augmentations
        augmented_data = self.transform({
            'image': mixed_data['mixed_image'],
            'labels': mixed_labels
        })
        
        return {
            'mixed_image': augmented_data['image'],
            'mixed_labels': augmented_data['labels'],
            'source_image': source_data['image'],
            'source_labels': source_data['labels'],
            'progress_ratio': progress_ratio,
            'mixing_mask': mixed_data['mixing_mask']
        }
    
    def _get_random_source_data(self):
        """Lädt zufälliges Source-Bild mit Labels"""
        source_path = random.choice(self.source_images)
        label_path = self.source_labels / f"{source_path.stem}.txt"
        
        image = Image.open(source_path)
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                labels = f.readlines()
        
        return {
            'image': image,
            'labels': labels
        }
    
    def _generate_mixed_labels(self, source_labels, target_detections, selected_region, image_size):
        """Generiert kombinierte YOLO-Labels für gemischtes Bild"""
        mixed_labels = []
        x1, y1, x2, y2 = selected_region
        
        # Add target pseudo-labels in selected region
        for det in target_detections:
            box = det['box']
            box_center_x = (box[0] + box[2]) / 2
            box_center_y = (box[1] + box[3]) / 2
            
            if (x1 <= box_center_x <= x2 and y1 <= box_center_y <= y2):
                yolo_box = YOLOUtils.convert_to_yolo_format(
                    box, image_size[0], image_size[1]
                )
                mixed_labels.append({
                    'class': det['class'],
                    'box': yolo_box
                })
        
        # Add source labels outside selected region
        for label in source_labels:
            parts = label.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * image_size[0]
                y_center = float(parts[2]) * image_size[1]
                
                if not (x1 <= x_center <= x2 and y1 <= y_center <= y2):
                    mixed_labels.append({
                        'class': class_id,
                        'box': (float(parts[1]), float(parts[2]), 
                               float(parts[3]), float(parts[4]))
                    })
        
        return mixed_labels
    
    def _get_yolo_augmentations(self):
        """YOLO-spezifische Augmentationen"""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                size=(640, 640),
                scale=(0.8, 1.0)
            )
        ])

def create_confmix_dataloader(source_path, target_path, confmix_detector, 
                            batch_size=8, num_workers=0):
    """Erstellt DataLoader mit ConfMix Dataset"""
    dataset = ConfMixDataset(source_path, target_path, confmix_detector)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_confmix_batch
    )
def collate_confmix_batch(batch):
    """Custom collate function für ConfMix batches"""
    return {
        'mixed_images': torch.stack([transforms.ToTensor()(item['mixed_image']) for item in batch]),
        'mixed_labels': [item['mixed_labels'] for item in batch],
        'source_images': torch.stack([transforms.ToTensor()(item['source_image']) for item in batch]),
        'source_labels': [item['source_labels'] for item in batch],
        'progress_ratio': torch.tensor([item['progress_ratio'] for item in batch]),
        'mixing_masks': torch.stack([torch.from_numpy(item['mixing_mask']) for item in batch])
    }
