"""
Surgical Workflow Verb Recognition Dataset

This module provides dataset and dataloader implementations for training models to recognize
surgical actions (verbs) from video frames. It handles loading and preprocessing of surgical
workflow images, with support for multi-label classification of surgical instruments and actions.

Features:
- Custom dataset implementation for surgical workflow images
- Support for instrument-verb constraints via adjacency matrix
- Balanced sampling for handling class imbalance
- Data augmentation pipeline for training
- Train/validation split functionality

Example usage:
    ```python
    data_module = VerbDataModule(base_dir="/path/to/data")
    data_module.setup()
    train_loader = data_module.train_dataloader()
    ```
"""

import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import pytorch_lightning as pl
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import Counter
from tqdm import tqdm


class VerbDataset(Dataset):
    """
    Dataset class for loading surgical workflow images with corresponding verb and instrument labels.
    
    The dataset expects images named in the format:
    [frame_number]_[instrument]_[verb]_conf[confidence].png
    
    Attributes:
        base_dir (str): Root directory containing the image data
        transform (transforms.Compose, optional): Image transformations to apply
        phase (str): Dataset phase ('train' or 'val')
        verb_names (List[str]): List of valid surgical verbs
        instrument_names (List[str]): List of valid surgical instruments
        instrument_verb_matrix (torch.Tensor): Boolean matrix defining valid instrument-verb combinations
    """
    
    def __init__(self, base_dir: str, transform: Optional[transforms.Compose] = None, phase: str = 'train'):
        """
        Initialize the VerbDataset.
        
        Args:
            base_dir: Root directory containing the image data
            transform: Optional transforms to be applied to images
            phase: Dataset phase ('train' or 'val')
        """
        self.base_dir = base_dir
        self.transform = transform
        self.phase = phase
        
        # Define verb and instrument classes
        self.verb_names = [
            "dissect", "retract", "null_verb", "coagulate", "grasp",
            "clip", "aspirate", "cut", "irrigate"
        ]
        self.instrument_names = [
            "grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"
        ]
        
        # Create mapping dictionaries for labels
        self.verb_to_idx = {verb: idx for idx, verb in enumerate(self.verb_names)}
        self.instrument_to_idx = {inst: idx for idx, inst in enumerate(self.instrument_names)}
        
        # Define constraint matrix for valid instrument-verb combinations
        self.instrument_verb_matrix = torch.tensor([
            #grasp  bipol  hook  sciss  clip  irrig
            [0,     0,     1,    1,     0,    0],  # dissect
            [1,     1,     1,    0,     0,    1],  # retract
            [1,     1,     1,    1,     1,    1],  # null_verb
            [0,     1,     1,    0,     0,    0],  # coagulate
            [1,     1,     0,    0,     0,    0],  # grasp
            [0,     0,     0,    0,     1,    0],  # clip
            [0,     0,     0,    0,     0,    1],  # aspirate
            [0,     0,     0,    1,     0,    0],  # cut
            [0,     0,     0,    0,     0,    1],  # irrigate
        ], dtype=torch.bool)
        
        self.data = self.load_data()

    def parse_filename(self, filename: str) -> Optional[Dict]:
        """
        Extract information from filename following the pattern:
        [frame_number]_[instrument]_[verb]_conf[confidence].png
        
        Args:
            filename: Name of the image file to parse
            
        Returns:
            Dictionary containing parsed information or None if parsing fails
        """
        try:
            base = filename.rsplit('.', 1)[0]
            parts = base.split('_')
            
            frame_number = int(parts[0])
            instrument = parts[1]
            
            # Handle null_verb special case
            if len(parts) == 5 and parts[2] == "null" and parts[3] == "verb":
                verb = "null_verb"
                confidence = float(parts[-1].replace('conf', ''))
            else:
                verb = parts[2]
                confidence = float(parts[-1].replace('conf', ''))
            
            # Validate parsed information
            if verb not in self.verb_names or instrument not in self.instrument_names:
                return None
                
            verb_idx = self.verb_to_idx[verb]
            inst_idx = self.instrument_to_idx[instrument]
            
            # Check constraint matrix
            if not self.instrument_verb_matrix[verb_idx, inst_idx]:
                return None
            
            return {
                'frame': frame_number,
                'instrument': instrument,
                'verb': verb,
                'verb_idx': verb_idx,
                'confidence': confidence
            }
        except Exception as e:
            print(f"Error parsing filename {filename}: {e}")
            return None

    def load_data(self) -> List[Dict]:
        """
        Load all valid images and their metadata from the dataset directory.
        
        Returns:
            List of dictionaries containing image metadata
        """
        all_data = []
        
        for vid_dir in tqdm(os.listdir(self.base_dir), desc="Loading data"):
            if not vid_dir.startswith('VID'):
                continue
            
            vid_path = os.path.join(self.base_dir, vid_dir)
            
            for filename in os.listdir(vid_path):
                if not filename.endswith('.png'):
                    continue
                    
                file_info = self.parse_filename(filename)
                if file_info is None:
                    continue
                
                img_path = os.path.join(vid_path, filename)
                if os.path.exists(img_path):
                    file_info['img_path'] = img_path
                    file_info['vid'] = vid_dir
                    file_info['filename'] = filename
                    all_data.append(file_info)
        
        if not all_data:
            raise RuntimeError("No data loaded!")
            
        print(f"Loaded {len(all_data)} images total")
        
        # Debug information
        print("\nExample filename parsing:")
        for i, item in enumerate(all_data[:5]):
            print(f"\nImage {i+1}:")
            print(f"Filename: {item['filename']}")
            print(f"Verb: {item['verb']} (idx: {item['verb_idx']})")
            print(f"Instrument: {item['instrument']}")
            print(f"Confidence: {item['confidence']}")
        
        return all_data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            Tuple containing (image_tensor, instrument_name, verb_index)
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of valid range (0-{len(self.data)-1})")
            
        item = self.data[idx]
        
        try:
            image = Image.open(item['img_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, item['instrument'], item['verb_idx']
            
        except Exception as e:
            raise Exception(f"Error loading image {item['img_path']}: {e}")

    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.data)


class VerbDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling surgical workflow data.
    
    This module handles:
    - Data loading and preprocessing
    - Train/validation splits
    - Data augmentation
    - Balanced sampling for handling class imbalance
    """
    
    def __init__(self, base_dir: str, batch_size: int = 32, num_workers: int = 4):
        """
        Initialize the DataModule.
        
        Args:
            base_dir: Root directory containing the image data
            batch_size: Number of samples per batch
            num_workers: Number of worker processes for data loading
        """
        super().__init__()
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define training augmentations
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Define validation transforms
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def setup(self, stage: Optional[str] = None):
        """
        Set up train and validation datasets.
        
        Args:
            stage: Current stage ('fit' or 'test')
        """
        if stage == 'fit' or stage is None:
            self.full_dataset = VerbDataset(
                self.base_dir, 
                self.train_transform, 
                'train'
            )
            
            # Create train/validation split
            dataset_size = len(self.full_dataset)
            indices = list(range(dataset_size))
            train_size = int(0.85 * dataset_size)
            
            np.random.shuffle(indices)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            self.train_dataset = Subset(self.full_dataset, train_indices)
            self.val_dataset = Subset(self.full_dataset, val_indices)
            
            # Calculate sampling weights for balanced training
            verb_instrument_pairs = [
                (self.full_dataset.data[i]['verb'], self.full_dataset.data[i]['instrument'])
                for i in train_indices
            ]
            pair_counts = Counter(verb_instrument_pairs)
            
            total_samples = len(verb_instrument_pairs)
            weights = [
                total_samples / (len(pair_counts) * pair_counts[pair])
                for pair in verb_instrument_pairs
            ]
            self.train_weights = torch.DoubleTensor(weights)

        # Print validation set statistics
        val_verbs = Counter([self.full_dataset.data[i]['verb'] for i in val_indices])
        print("Validation set verb distribution:", val_verbs)

    def train_dataloader(self) -> DataLoader:
        """
        Create the training data loader with balanced sampling.
        
        Returns:
            DataLoader for training
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=WeightedRandomSampler(
                weights=self.train_weights,
                num_samples=len(self.train_dataset),
                replacement=True
            ),
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create the validation data loader.
        
        Returns:
            DataLoader for validation
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


if __name__ == "__main__":
    # Example usage
    base_dir = "/data/Bartscht/Verbs" 
    data_module = VerbDataModule(base_dir)
    data_module.setup()
    
    # Test the dataloader
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    images, instrument_names, verb_labels = batch
    
    print("\nExample batch:")
    print(f"Images shape: {images.shape}")
    print(f"Instrument names (first 5): {instrument_names[:5]}")
    print(f"Verb labels (first 5): {verb_labels[:5]}")