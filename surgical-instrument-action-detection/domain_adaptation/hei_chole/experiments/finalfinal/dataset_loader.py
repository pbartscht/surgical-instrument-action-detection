import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms

class DetectionDataset(Dataset):
    def __init__(self, path, with_labels=True, video_structure=False):
        self.path = Path(path)
        self.with_labels = with_labels
        self.image_size = (640, 640)  # YOLO standard size
        
        # Einfache Transformation: Nur resize auf YOLO-Größe und Konvertierung zu Tensor
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),  # Dies konvertiert auch zu [0-1] Range
        ])
        
        # Setup Datenpfade
        if video_structure:
            self.image_files = []
            for vid_dir in (self.path / "Videos").glob("VID*"):
                self.image_files.extend(list(vid_dir.glob("*.png")))
        else:
            self.image_files = list((self.path / "images" / "train").glob("*.png"))
            if with_labels:
                self.label_path = self.path / "labels" / "train"

        if not self.image_files:
            raise RuntimeError(f"No images found in {self.path}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load and process image
        image_path = self.image_files[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image_tensor = torch.zeros(3, *self.image_size)

        # Initialize empty labels list
        labels = []

        # Load labels if available
        if self.with_labels:
            label_file = self.label_path / f"{image_path.stem}.txt"
            if label_file.exists():
                try:
                    with open(label_file) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:  # class, x, y, w, h format
                                class_id = int(parts[0])
                                bbox = torch.tensor([float(x) for x in parts[1:]])
                                labels.append({
                                    'class_id': class_id,
                                    'bbox': bbox
                                })
                except Exception as e:
                    print(f"Error loading labels from {label_file}: {e}")

        return {
            'image': image_tensor,
            'labels': labels,
            'path': str(image_path),
            'image_id': idx
        }

def detection_collate_fn(batch):
    """
    Custom collate function for detection batches
    """
    images = torch.stack([item['image'] for item in batch])
    paths = [item['path'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    
    # Collect all labels
    all_labels = []
    max_labels = max(len(item['labels']) for item in batch)
    
    for item in batch:
        labels = item['labels']
        # Pad with dummy labels if necessary
        while len(labels) < max_labels:
            labels.append({
                'class_id': -1,  # Use -1 as padding class
                'bbox': torch.zeros(4)  # Zero bbox for padding
            })
        all_labels.append(labels)
    
    # Convert labels to tensors
    if max_labels > 0:
        class_ids = torch.tensor([[l['class_id'] for l in item_labels] for item_labels in all_labels])
        bboxes = torch.stack([torch.stack([l['bbox'] for l in item_labels]) for item_labels in all_labels])
    else:
        class_ids = torch.empty((len(batch), 0), dtype=torch.long)
        bboxes = torch.empty((len(batch), 0, 4), dtype=torch.float)
    
    return {
        'images': images,
        'class_ids': class_ids,
        'bboxes': bboxes,
        'paths': paths,
        'image_ids': image_ids
    }

def create_dataloaders(source_path, target_path, batch_size=8, num_workers=8):
    """Create separate source and target dataloaders"""
    source_dataset = DetectionDataset(
        source_path, 
        with_labels=True,
    )
    
    target_dataset = DetectionDataset(
        target_path, 
        with_labels=False,
        video_structure=True
    )
    
    print(f"Created datasets - Source size: {len(source_dataset)}, Target size: {len(target_dataset)}")
    
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    return source_loader, target_loader