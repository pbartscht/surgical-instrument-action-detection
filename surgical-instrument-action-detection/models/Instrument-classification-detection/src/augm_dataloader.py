import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset
import albumentations as A
import torch

class BasicSurgicalYOLODataset(YOLODataset):
    """
    Custom YOLO dataset for surgical instrument detection with specialized medical augmentations.
    Extends the base YOLODataset with augmentations specifically designed for surgical scenarios.
    
    Features:
    - Basic augmentations for surgical environments (brightness, contrast, geometric)
    - Medical-specific augmentations (CLAHE for metallic instruments, shadow simulation)
    - Memory-efficient processing
    - Fallback handling for augmentation failures
    """

    def __init__(self, data_path, augment=False, class_weights=None, **kwargs):
        """
        Initialize the surgical dataset with specialized augmentations.
        
        Args:
            *args: Variable length argument list passed to YOLODataset
            **kwargs: Arbitrary keyword arguments passed to YOLODataset
        """
        super().__init__(data_path, augment=augment, **kwargs)
        self.class_weights = class_weights

        # Stage 1: Essential base augmentations for surgical instruments
        basic_transforms = [
            # Brightness and contrast adjustment for varying OR lighting conditions
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.2, 0.2),
                p=0.8  # High probability as it's crucial for OR scenes
            ),

            # Geometric transformations for various instrument positions
            A.ShiftScaleRotate(
                shift_limit=0.1,    # Subtle position shifts
                scale_limit=0.1,    # Scaling for depth variations
                rotate_limit=20,    # Rotation for different grip angles
                border_mode=cv2.BORDER_CONSTANT,
                p=0.7
            ),
        ]

        # Stage 2: Medical-specific augmentations
        medical_transforms = [
            # CLAHE for better instrument contours
            # Essential for metallic instruments under OR lighting
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(4, 4),
                p=0.3
            ),

            # Shadow simulation for realistic OR conditions
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

    def __getitem__(self, index):
        """
        Get a single sample from the dataset with augmentations applied.
        
        Args:
            index (int): Index of the sample to fetch
            
        Returns:
            tuple: (image, labels) where image is the augmented image and 
                labels are the corresponding bounding box annotations
        """
        try:
            print(f"DEBUG: Method called with index={index}, augment={self.augment}")
            # Get the original image and labels from parent class
            im, labels = super().__getitem__(index)
            print(f"Debug - augment: {self.augment}, labels: {len(labels)}")
            print(f"Debug - image type: {type(im)}, shape: {im.shape}")
            
            if self.augment and len(labels):
                print(f"DEBUG: Applying custom augmentations to image {index}")
                # Convert image to correct format if needed (ultralytics might return different format)
                if isinstance(im, torch.Tensor):
                    im = im.numpy()
                if im.shape[0] == 3:  # If channels are first, transpose to HWC
                    im = im.transpose(1, 2, 0)
                
                # Apply augmentations
                augmented = self.instrument_aug(
                    image=im,
                    bboxes=labels[:, 1:].tolist(),  # Convert bbox coordinates to list
                    class_labels=labels[:, 0].tolist()  # Convert class labels to list
                )
                print(f"DEBUG: Augmentation applied successfully")

                # Update image with augmented version
                im = augmented['image']

                # Reconstruct labels if bounding boxes exist
                if augmented['bboxes']:
                    labels = np.array(
                        [[c, *b] for c, b in zip(
                            augmented['class_labels'],
                            augmented['bboxes']
                        )],
                        dtype=np.float32
                    )

                # Convert back to torch tensor if needed
                if isinstance(super().__getitem__(index)[0], torch.Tensor):
                    im = torch.from_numpy(im)
                    if len(im.shape) == 3:
                        im = im.permute(2, 0, 1)  # HWC to CHW

                # Explicitly free memory
                del augmented
                
                print(f"DEBUG: Returning augmented image shape={im.shape}, labels shape={labels.shape}")

            return im, labels

        except Exception as e:
            print(f"ERROR in __getitem__: {str(e)}")
            print(f"Falling back to unaugmented sample for index {index}")
            return super().__getitem__(index)