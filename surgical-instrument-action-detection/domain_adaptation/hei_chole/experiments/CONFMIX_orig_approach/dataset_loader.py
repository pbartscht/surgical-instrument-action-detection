import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
from torchvision import transforms
import random

class UDADataset(Dataset):
    def __init__(self, 
                 path, 
                 img_size=640,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 is_source=True,
                 is_video_structure=False,
                 stride=32,
                 pad=0.0):
        """Initialize UDA Dataset"""
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = rect
        self.is_source = is_source
        self.stride = stride
        self.path = Path(path)

        # Initialize file paths
        try:
            if is_video_structure:
                # For video structure (target domain)
                self.img_files = []
                for vid_dir in (self.path / "Videos").glob("VID*"):
                    self.img_files.extend(list(vid_dir.glob("*.png")))
            else:
                # For standard YOLO structure (source domain)
                self.img_files = list((self.path / "images" / "train").glob("*.png"))
                if is_source:
                    self.label_files = [
                        self.path / "labels" / "train" / f"{f.stem}.txt"
                        for f in self.img_files
                    ]

            self.img_files = sorted([str(f) for f in self.img_files])  # Sort files
            assert len(self.img_files) > 0, f'No images found in {path}'

        except Exception as e:
            raise Exception(f'Error loading data from {path}: {e}')

        # Cache labels
        if self.is_source:
            self.labels = [np.zeros((0, 5), dtype=np.float32)] * len(self.img_files)
            for i, label_file in enumerate(self.label_files):
                try:
                    if os.path.exists(label_file):
                        with open(label_file, 'r') as f:
                            labels = [x.split() for x in f.read().strip().splitlines()]
                            if labels:
                                self.labels[i] = np.array(labels, dtype=np.float32)
                except Exception as e:
                    print(f'Warning: Error loading {label_file}: {e}')

        # Handle rectangular training
        if self.rect:
            shapes = [self._get_image_shape(f) for f in self.img_files]
            self.shapes = np.array(shapes, dtype=np.float64)
            
            # Calculate aspect ratio and sort
            ar = self.shapes[:, 1] / self.shapes[:, 0]  # aspect ratio
            self.irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in self.irect]
            if self.is_source:
                self.labels = [self.labels[i] for i in self.irect]
                self.shapes = self.shapes[self.irect]

            # Set training image shapes
            shapes = [[1, 1]] * len(self.shapes)
            for i in range(len(self.shapes)):
                shapes[i] = [
                    max(self.img_size[0], int(self.shapes[i][0] / self.stride + pad) * self.stride),
                    max(self.img_size[1], int(self.shapes[i][1] / self.stride + pad) * self.stride)
                ]
            self.batch_shapes = np.array(shapes, dtype=np.int64)

        # Setup transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # Load image
        img_path = self.img_files[index]
        img = self.load_image(img_path)
        h0, w0 = img.shape[:2]  # orig hw
        
        # Resize
        r = self.img_size[0] / max(h0, w0)  # resize image to img_size
        if r != 1:  
            interp = cv2.INTER_LINEAR if self.augment else cv2.INTER_AREA
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]  # current hw

        # Load labels for source domain
        labels = []
        if self.is_source:
            labels = self.labels[index].copy()
            # Normalized xywh to pixel xyxy format
            if len(labels):
                labels[:, 1:] = self._xywhn2xyxy(labels[:, 1:], w0, h0, padw=0, padh=0)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        
        if self.is_source:
            labels = torch.from_numpy(labels)
        else:
            labels = torch.zeros((0, 5))  # Empty labels for target domain

        return img, labels, img_path, (h0, w0)

    def load_image(self, path):
        """Load and preprocess image"""
        img = cv2.imread(path)  # BGR
        assert img is not None, f'Image Not Found {path}'
        return img

    @staticmethod
    def _get_image_shape(path):
        """Get image shape without loading"""
        im = cv2.imread(path)
        return im.shape[:2] if im is not None else (0, 0)

    @staticmethod
    def _xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
        """Convert normalized xywh to pixel xyxy format"""
        y = np.copy(x)
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
        return y

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader"""
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

def create_uda_dataloaders(source_path,
                          target_path,
                          img_size=640,
                          batch_size=16,
                          stride=32,
                          hyp=None,
                          augment=True,
                          rect=False,
                          workers=8,
                          image_weights=False,
                          prefix=''):
    """Create dataloaders for unsupervised domain adaptation"""
    
    # Initialize Datasets
    dataset_s = UDADataset(
        path=source_path,
        img_size=img_size,
        augment=augment,
        hyp=hyp,
        rect=rect,
        image_weights=image_weights,
        is_source=True,
        is_video_structure=False,
        stride=stride
    )
    
    dataset_t = UDADataset(
        path=target_path,
        img_size=img_size,
        augment=augment,
        hyp=hyp,
        rect=rect,
        image_weights=image_weights,
        is_source=False,
        is_video_structure=True,
        stride=stride
    )
    
    batch_size = min(batch_size, len(dataset_s), len(dataset_t))
    workers = min(workers, os.cpu_count() - 1) if os.cpu_count() else workers
    
    # Create Dataloaders
    loader_s = DataLoader(
        dataset_s,
        batch_size=batch_size,
        shuffle=True and not image_weights,
        num_workers=workers,
        pin_memory=True,
        collate_fn=UDADataset.collate_fn
    )
    
    loader_t = DataLoader(
        dataset_t,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=UDADataset.collate_fn
    )
    
    return loader_s, dataset_s, loader_t, dataset_t