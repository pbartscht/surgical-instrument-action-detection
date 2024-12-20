"""
CholecT50 Dataset and DataLoader Implementation

This module provides PyTorch dataset classes and data loading utilities for the CholecT50 dataset,
which contains surgical video data with various annotations including tools, actions, and phases.

The implementation is based on the original work by: 
C.I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux, N. Padoy. Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. Medical Image Analysis, 78 (2022) 102433.

with modifications for improved Instrument-Verb-Pairs datalaoding

Classes:
    CholecT50_DataModule: PyTorch Lightning DataModule for CholecT50
    CholecT50: Main dataset class handling data splitting and preparation
    T50: Base dataset class for individual video handling

"""

import os
import json
from typing import Tuple, List, Dict, Optional, Any
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import pytorch_lightning as pl
import torchvision.transforms as transforms

class CholecT50_DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for CholecT50 dataset."""
    
    def __init__(
        self, 
        dataset_dir: str, 
        batch_size: int,
        dataset_variant: str = "cholect45-crossval"
    ):
        """
        Initialize the DataModule.

        Args:
            dataset_dir: Root directory of the CholecT50 dataset
            batch_size: Batch size for dataloaders
            dataset_variant: Dataset variant to use (default: "cholect45-crossval")
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_variant = dataset_variant
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up the datasets for training, validation, and testing.
        
        Args:
            stage: Optional stage parameter (fit/test)
        """
        cholect50 = CholecT50(
            dataset_dir=self.dataset_dir, 
            dataset_variant=self.dataset_variant,
            img_size=(256, 448)
        )
        
        self.train_dataset, self.val_dataset, self.test_dataset = cholect50.build()
        
    def train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

class CholecT50:
    """Main CholecT50 dataset class handling data splitting and preparation."""
    
    DATASET_VARIANTS = {
        "cholect45-crossval": "CholecT45 dataset variant with official cross-validation splits",
        "cholect50-crossval": "CholecT50 dataset variant with official cross-validation splits (recommended)",
        "cholect50-challenge": "CholecT50 dataset variant used in CholecTriplet challenge",
        "cholect50": "CholecT50 dataset with original splits from rendezvous paper",
        "cholect45": "Pointer to cholect45-crossval",
        "cholect50-subset": "Subset created for EDU4SDS summer school"
    }

    def __init__(
        self, 
        dataset_dir: str, 
        dataset_variant: str = "cholect50-challenge",
        img_size: Tuple[int, int] = (224, 224),
        test_fold: int = 1,
        augmentation_list: List[str] = None,
        normalize: bool = True
    ):
        """
        Initialize the CholecT50 dataset.

        Args:
            dataset_dir: Root directory of the dataset
            dataset_variant: Which dataset variant to use
            img_size: Size to resize images to (height, width)
            test_fold: Which fold to use for testing in cross-validation
            augmentation_list: List of augmentation names to apply
            normalize: Whether to normalize images
        
        Raises:
            ValueError: If dataset_variant is not valid
        """
        if dataset_variant not in self.DATASET_VARIANTS:
            raise ValueError(
                f"Invalid dataset variant: {dataset_variant}. "
                f"Valid options are: {list(self.DATASET_VARIANTS.keys())}"
            )
            
        self.img_size = img_size
        self.normalize = normalize
        self.dataset_dir = Path(dataset_dir)
        
        # Set up data splits
        video_split = self._get_video_split(dataset_variant)
        self.train_records, self.val_records, self.test_records = \
            self._prepare_record_lists(video_split, test_fold, dataset_variant)
            
        # Set up augmentations
        if augmentation_list is None:
            augmentation_list = ['original', 'vflip', 'hflip', 'contrast', 'rot90']
        self.augmentation_list = self._get_augmentations(augmentation_list)
        
        # Build transforms
        trainform, testform = self._build_transforms()
        
        # Build datasets
        self._build_datasets(trainform, testform)

    def _get_augmentations(self, augmentation_list: List[str]) -> List[transforms.Transform]:
        """Get list of augmentation transforms based on names."""
        aug_dict = {
            'original': transforms.Lambda(lambda x: x),
            'vflip': transforms.RandomVerticalFlip(0.4),
            'hflip': transforms.RandomHorizontalFlip(0.4),
            'contrast': transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
            'rot90': transforms.RandomRotation(90, expand=True),
            'brightness': transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.5),
            'autocontrast': transforms.RandomAutocontrast(p=0.5),
        }
        
        return [aug_dict[aug] for aug in augmentation_list if aug in aug_dict]

    def _build_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Build training and testing transforms."""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        basic_transforms = [transforms.Resize(self.img_size)]
        test_transforms = basic_transforms + [transforms.ToTensor()]
        train_transforms = basic_transforms + self.augmentation_list + [transforms.ToTensor()]
        
        if self.normalize:
            test_transforms.append(normalize)
            train_transforms.append(normalize)
            
        return transforms.Compose(train_transforms), transforms.Compose(test_transforms)

    def _build_datasets(self, trainform: transforms.Compose, testform: transforms.Compose) -> None:
        """Build train, validation and test datasets."""
        def build_dataset(video_list: List[str], transform: transforms.Compose) -> ConcatDataset:
            datasets = []
            for video in video_list:
                dataset = T50(
                    img_dir=self.dataset_dir / 'videos' / video,
                    label_file=self.dataset_dir / 'labels' / f'{video}.json',
                    transform=transform
                )
                datasets.append(dataset)
            return ConcatDataset(datasets)
            
        self.train_dataset = build_dataset(self.train_records, trainform)
        self.val_dataset = build_dataset(self.val_records, trainform)
        self.test_dataset = build_dataset(self.test_records, testform)

    def build(self) -> Tuple[ConcatDataset, ConcatDataset, ConcatDataset]:
        """Get the built datasets."""
        return self.train_dataset, self.val_dataset, self.test_dataset

    @staticmethod
    def _get_video_split(variant: str) -> Dict:
        """Get video split configuration for given variant."""
        # Video split configurations...
        splits = {
            'cholect50': {
                'train': [1, 15, 26, 40, 52, 65, 79, 2, 18, 27, 43, 56, 66, 92],
                'val': [8, 12, 29, 50, 78],
                'test': [6, 51, 10, 73, 14, 74, 32, 80, 42, 111]
            },
            # Add other split configurations...
        }
        return splits[variant]

class T50(Dataset):
    def __init__(self, img_dir, label_file, transform=None, target_transform=None):
        label_data = json.load(open(label_file, "rb"))
        self.label_data = label_data["annotations"]
        self.frames = list(self.label_data.keys())
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.iv_map = self.create_iv_map()
        self.iv_real_map = self.create_iv_real_map()
    
    def create_iv_map(self):
        map_data = self.map_file()
        iv_map = {}
        for row in map_data:
            ivt, i, v, t, iv, it = row
            iv_map[ivt] = iv
        return iv_map
    
    def create_iv_real_map(self):
        map_data = self.map_file()
        used_ivs = sorted(set(row[4] for row in map_data))
        return {iv: i for i, iv in enumerate(used_ivs)}
    
    
    def map_file(self):
        return np.array([
            [ 0,  0,  2,  1,  2,  1],
            [ 1,  0,  2,  0,  2,  0],
            [ 2,  0,  2, 10,  2, 10],
            [ 3,  0,  0,  3,  0,  3],
            [ 4,  0,  0,  2,  0,  2],
            [ 5,  0,  0,  4,  0,  4],
            [ 6,  0,  0,  1,  0,  1],
            [ 7,  0,  0,  0,  0,  0],
            [ 8,  0,  0, 12,  0, 12],
            [ 9,  0,  0,  8,  0,  8],
            [10,  0,  0, 10,  0, 10],
            [11,  0,  0, 11,  0, 11],
            [12,  0,  0, 13,  0, 13],
            [13,  0,  8,  0,  8,  0],
            [14,  0,  1,  2,  1,  2],
            [15,  0,  1,  4,  1,  4],
            [16,  0,  1,  1,  1,  1],
            [17,  0,  1,  0,  1,  0],
            [18,  0,  1, 12,  1, 12],
            [19,  0,  1,  8,  1,  8],
            [20,  0,  1, 10,  1, 10],
            [21,  0,  1, 11,  1, 11],
            [22,  1,  3,  7, 13, 22],
            [23,  1,  3,  5, 13, 20],
            [24,  1,  3,  3, 13, 18],
            [25,  1,  3,  2, 13, 17],
            [26,  1,  3,  4, 13, 19],
            [27,  1,  3,  1, 13, 16],
            [28,  1,  3,  0, 13, 15],
            [29,  1,  3,  8, 13, 23],
            [30,  1,  3, 10, 13, 25],
            [31,  1,  3, 11, 13, 26],
            [32,  1,  2,  9, 12, 24],
            [33,  1,  2,  3, 12, 18],
            [34,  1,  2,  2, 12, 17],
            [35,  1,  2,  1, 12, 16],
            [36,  1,  2,  0, 12, 15],
            [37,  1,  2, 10, 12, 25],
            [38,  1,  0,  1, 10, 16],
            [39,  1,  0,  8, 10, 23],
            [40,  1,  0, 13, 10, 28],
            [41,  1,  1,  2, 11, 17],
            [42,  1,  1,  4, 11, 19],
            [43,  1,  1,  0, 11, 15],
            [44,  1,  1,  8, 11, 23],
            [45,  1,  1, 10, 11, 25],
            [46,  2,  3,  5, 23, 35],
            [47,  2,  3,  3, 23, 33],
            [48,  2,  3,  2, 23, 32],
            [49,  2,  3,  4, 23, 34],
            [50,  2,  3,  1, 23, 31],
            [51,  2,  3,  0, 23, 30],
            [52,  2,  3,  8, 23, 38],
            [53,  2,  3, 10, 23, 40],
            [54,  2,  5,  5, 25, 35],
            [55,  2,  5, 11, 25, 41],
            [56,  2,  2,  5, 22, 35],
            [57,  2,  2,  3, 22, 33],
            [58,  2,  2,  2, 22, 32],
            [59,  2,  2,  1, 22, 31],
            [60,  2,  2,  0, 22, 30],
            [61,  2,  2, 10, 22, 40],
            [62,  2,  2, 11, 22, 41],
            [63,  2,  1,  0, 21, 30],
            [64,  2,  1,  8, 21, 38],
            [65,  3,  3, 10, 33, 55],
            [66,  3,  5,  9, 35, 54],
            [67,  3,  5,  5, 35, 50],
            [68,  3,  5,  3, 35, 48],
            [69,  3,  5,  2, 35, 47],
            [70,  3,  5,  1, 35, 46],
            [71,  3,  5,  8, 35, 53],
            [72,  3,  5, 10, 35, 55],
            [73,  3,  5, 11, 35, 56],
            [74,  3,  2,  1, 32, 46],
            [75,  3,  2,  0, 32, 45],
            [76,  3,  2, 10, 32, 55],
            [77,  4,  4,  5, 44, 65],
            [78,  4,  4,  3, 44, 63],
            [79,  4,  4,  2, 44, 62],
            [80,  4,  4,  4, 44, 64],
            [81,  4,  4,  1, 44, 61],
            [82,  5,  6,  6, 56, 81],
            [83,  5,  2,  2, 52, 77],
            [84,  5,  2,  4, 52, 79],
            [85,  5,  2,  1, 52, 76],
            [86,  5,  2,  0, 52, 75],
            [87,  5,  2, 10, 52, 85],
            [88,  5,  7,  7, 57, 82],
            [89,  5,  7,  4, 57, 79],
            [90,  5,  7,  8, 57, 83],
            [91,  5,  1,  0, 51, 75],
            [92,  5,  1,  8, 51, 83],
            [93,  5,  1, 10, 51, 85],
            [94,  0,  9, 14,  9, 14],
            [95,  1,  9, 14, 19, 29],
            [96,  2,  9, 14, 29, 44],
            [97,  3,  9, 14, 39, 59],
            [98,  4,  9, 14, 49, 74],
            [99,  5,  9, 14, 59, 89]
        ])
    
    def get_binary_labels(self, labels):
        iv_label = np.zeros(len(self.iv_real_map))
        tool_label = np.zeros([6])
        verb_label = np.zeros([10])
        target_label = np.zeros([15])
        phase_label = np.zeros([7])
        
        for label in labels:
            ivt = label[0]
            if ivt != -1.0:
                iv = self.iv_map.get(ivt, -1)
                if iv != -1:
                    iv_real = self.iv_real_map.get(iv, -1)
                    if iv_real != -1:
                        iv_label[iv_real] += 1
            
            tool = label[1:7]
            if tool[0] != -1.0:
                tool_label[tool[0]] += 1
            
            verb = label[7:8]
            if verb[0] != -1.0:
                verb_label[verb[0]] += 1
            
            target = label[8:14]
            if target[0] != -1.0:
                target_label[target[0]] += 1
            
            phase = label[14:15]
            if phase[0] != -1.0:
                phase_label[phase[0]] += 1
        
        #print(f"IV label shape: {iv_label.shape}") 

        return (iv_label, tool_label, verb_label, target_label, phase_label)
    
    def __getitem__(self, index):
        labels = self.label_data[self.frames[index]]
        basename = "{}.png".format(str(self.frames[index]).zfill(6))
        img_path = os.path.join(self.img_dir, basename)
        image = Image.open(img_path)
        labels = self.get_binary_labels(labels)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        
        return image, labels

    def __len__(self):
        return len(self.frames)
    

if __name__ == "__main__":
    PATH_TO_CHOLECT50 = "/data/Bartscht/CholecT50"
    cholect50 = CholecT50(dataset_dir=PATH_TO_CHOLECT50, augmentation_list=[], normalize=False)
    train_dataset, val_dataset, test_dataset = cholect50.build()