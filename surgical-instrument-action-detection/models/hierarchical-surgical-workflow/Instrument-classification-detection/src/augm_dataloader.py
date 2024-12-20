import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset
import albumentations as A
import torch
import yaml
from pathlib import Path

class BasicSurgicalYOLODataset(YOLODataset):
    """
    Custom YOLO dataset for surgical instrument detection with specialized medical augmentations.
    
    This dataset extends the base YOLO dataset with medical-specific image augmentations
    designed for surgical scenes and instrument detection.
    
    Attributes:
        class_weights (torch.Tensor, optional): Weights for each class for loss calculation
        instrument_aug (albumentations.Compose): Augmentation pipeline for surgical instruments
    """
    
    def __init__(self, data_path, augment=False, class_weights=None):
        """
        Initialize the surgical dataset with custom augmentations.
        
        Args:
            data_path (str): Path to the YAML file containing dataset configuration
            augment (bool): Whether to apply augmentations
            class_weights (torch.Tensor, optional): Class weights for loss calculation
        """
        # Load YAML configuration
        with open(data_path, 'r') as f:
            data_dict = yaml.safe_load(f)
            
        # Construct absolute path to training images
        base_path = Path(data_dict.get('path', ''))
        train_path = base_path / data_dict.get('train', '')
        
        print(f"Using image path: {train_path}")
        
        # Initialize parent YOLO dataset with absolute image path
        super().__init__(str(train_path), augment=augment)
        self.class_weights = class_weights
        
        if augment:
            self._setup_augmentations()
        else:
            self.instrument_aug = None
            print("Augmentations disabled")
    
    def _setup_augmentations(self):
        """
        Set up the augmentation pipeline with medical-specific transformations.
        Includes both basic augmentations and specialized medical image enhancements.
        """
        # Basic transformations for surgical scenes
        basic_transforms = [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.2, 0.2),
                p=0.8
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=20,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.7
            ),
        ]
        
        # Medical-specific transformations
        medical_transforms = [
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(4, 4),
                p=0.3
            ),
            A.RandomShadow(
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=3,
                p=0.3
            ),
        ]
        
        # Combine all transformations
        self.instrument_aug = A.Compose(
            basic_transforms + medical_transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            )
        )
        print("Augmentations initialized")
    
    def __getitem__(self, index):
        """
        Load and augment an image and its labels.
        
        Args:
            index (int): Index of the image to load
            
        Returns:
            tuple: (image, labels) where labels are in YOLO format
                  [class_id, x_center, y_center, width, height]
        """
        # Load original image and labels from YOLO
        img, labels = super().__getitem__(index)
        
        # Apply augmentations if enabled
        if self.augment and self.instrument_aug is not None:
            try:
                # Prepare labels for Albumentations
                bboxes = labels[:, 1:].tolist()  # YOLO format: [x_center, y_center, width, height]
                class_labels = labels[:, 0].tolist()
                
                # Apply augmentations
                transformed = self.instrument_aug(
                    image=img,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                # Create new labels matrix
                if transformed['bboxes']:
                    new_boxes = np.array(transformed['bboxes'])
                    new_labels = np.array(transformed['class_labels'])
                    labels = np.column_stack((new_labels, new_boxes))
                
                return transformed['image'], labels
                
            except Exception as e:
                print(f"Warning: Augmentation failed for image {index}: {str(e)}")
                return img, labels  # Fallback to non-augmented data
                
        return img, labels