#!/usr/bin/env python3
"""
CVAT Active Learning Integration Tool for Surgical Instrument Detection
===================================================================

This script integrates improved annotations from CVAT (Computer Vision Annotation Tool)
back into the training dataset as part of an active learning pipeline for surgical
instrument detection. It processes the corrected annotations from VID26 and
reorganizes the dataset for the next training iteration.

The script is part of a larger active learning workflow:
1. Initial model training
2. Model prediction on new data
3. Manual correction of predictions in CVAT
4. Integration of improved annotations (this script)
5. Retraining for improved model performance

Features:
- Processes corrected CVAT annotations from zip file
- Maintains dataset organization for YOLO training
- Creates standardized train/validation/test splits
- Preserves image-label pairs integrity
- Generates YOLO-compatible dataset structure
"""

import os
import shutil
import random
from zipfile import ZipFile
from typing import List, Tuple
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def organize_video_data(
    video_path: str = '/data/Bartscht/CholecT50/videos/VID26',
    annotations_zip: str = '/home/Bartscht/YOLO/surgical-instrument-action-detection/models/Instrument-classification-detection/outputs/predictions/cvat_annotations.zip',
    yolo_base_dir: str = '/data/Bartscht/YOLO',
    train_ratio: float = 0.7,
    val_ratio: float = 0.2
) -> None:
    """
    Organizes video data from CholecT50 dataset into YOLO format with train/val/test splits.
    Automatically adds video prefix to filenames based on source directory.

    Args:
        video_path (str): Path to the directory containing video images
        annotations_zip (str): Path to the zip file containing label annotations
        yolo_base_dir (str): Base directory where YOLO dataset will be created
        train_ratio (float): Ratio of data to use for training (0.0-1.0)
        val_ratio (float): Ratio of data to use for validation (0.0-1.0)
    """
    # Extract video prefix from path
    video_prefix = os.path.basename(video_path)
    logger.info(f"Processing video: {video_prefix}")
    
    # Input validation
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("Train and validation ratios must sum to less than 1.0")
    
    if not all(os.path.exists(p) for p in [video_path, annotations_zip]):
        raise FileNotFoundError("Input paths do not exist")

    # Create temporary directory for extracted labels
    temp_label_dir = f'/tmp/{video_prefix}_labels'
    os.makedirs(temp_label_dir, exist_ok=True)

    # Create YOLO directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(yolo_base_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(yolo_base_dir, 'labels', split), exist_ok=True)

    # Extract labels
    logger.info("Extracting labels...")
    with ZipFile(annotations_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_label_dir)

    label_dir = os.path.join(temp_label_dir, 'obj_train_data')

    # Collect valid image-label pairs
    logger.info("Collecting files...")
    valid_pairs = collect_valid_pairs(video_path, label_dir, video_prefix)
    
    # Split dataset
    train_pairs, val_pairs, test_pairs = split_dataset(
        valid_pairs, train_ratio, val_ratio
    )

    # Copy files to respective directories
    logger.info("Copying files...")
    stats = {
        'train': copy_pairs(train_pairs, 'train', yolo_base_dir),
        'val': copy_pairs(val_pairs, 'val', yolo_base_dir),
        'test': copy_pairs(test_pairs, 'test', yolo_base_dir)
    }

    # Update dataset files
    logger.info("Updating file lists...")
    update_dataset_files(yolo_base_dir, train_pairs, val_pairs, test_pairs)

    # Cleanup
    logger.info("Cleaning up...")
    shutil.rmtree(temp_label_dir)

    # Print statistics
    print_statistics(stats, len(valid_pairs))

def collect_valid_pairs(video_path: str, label_dir: str, video_prefix: str) -> List[Tuple[str, str, str, str]]:
    """
    Collects valid image-label pairs from the dataset.

    Returns:
        List of tuples containing (image_path, label_path, new_img_name, new_label_name)
    """
    valid_pairs = []
    for img_file in sorted(os.listdir(video_path)):
        if img_file.endswith('.png'):
            base_name = os.path.splitext(img_file)[0]
            label_file = f"{base_name}.txt"
            label_path = os.path.join(label_dir, label_file)
            
            # Create new filenames with prefix
            new_img_name = f"{video_prefix}_{img_file}"
            new_label_name = f"{video_prefix}_{label_file}"
            
            if os.path.exists(label_path):
                valid_pairs.append((
                    os.path.join(video_path, img_file),
                    label_path,
                    new_img_name,
                    new_label_name
                ))
            else:
                logger.warning(f"No label found for {img_file}")
    
    return valid_pairs

def split_dataset(valid_pairs: list, train_ratio: float, val_ratio: float) -> Tuple[list, list, list]:
    """
    Splits the dataset into train, validation, and test sets.

    Returns:
        Tuple of (train_pairs, val_pairs, test_pairs)
    """
    random.shuffle(valid_pairs)
    total = len(valid_pairs)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    return (
        valid_pairs[:train_size],
        valid_pairs[train_size:train_size + val_size],
        valid_pairs[train_size + val_size:]
    )

def copy_pairs(pairs: list, split: str, yolo_base_dir: str) -> int:
    """
    Copies image and label pairs to the appropriate split directory.

    Args:
        pairs: List of file pairs to copy
        split: Dataset split ('train', 'val', or 'test')
        yolo_base_dir: Base directory for YOLO dataset

    Returns:
        Number of pairs copied
    """
    copied = 0
    for img_path, label_path, new_img_name, new_label_name in pairs:
        img_dest = os.path.join(yolo_base_dir, 'images', split, new_img_name)
        label_dest = os.path.join(yolo_base_dir, 'labels', split, new_label_name)
        
        shutil.copy(img_path, img_dest)
        shutil.copy(label_path, label_dest)
        copied += 1
    
    return copied

def update_dataset_files(yolo_base_dir: str, train_pairs: list, val_pairs: list, test_pairs: list) -> None:
    """Updates the dataset files (train.txt, val.txt, test.txt, trainval.txt)"""
    def update_txt_file(filename: str, files_list: list) -> None:
        file_path = os.path.join(yolo_base_dir, filename)
        with open(file_path, 'a') as f:
            for _, _, new_img_name, _ in files_list:
                f.write(f'./images/{filename.replace(".txt", "")}/{new_img_name}\n')

    for split, pairs in [
        ('train.txt', train_pairs),
        ('val.txt', val_pairs),
        ('test.txt', test_pairs),
        ('trainval.txt', train_pairs + val_pairs)
    ]:
        update_txt_file(split, pairs)

def print_statistics(stats: dict, total: int) -> None:
    """Prints the dataset split statistics"""
    logger.info("\nIntegration completed:")
    logger.info(f"- Training images: {stats['train']}")
    logger.info(f"- Validation images: {stats['val']}")
    logger.info(f"- Test images: {stats['test']}")
    logger.info(f"Total: {sum(stats.values())} of {total} images processed")

if __name__ == "__main__":
    organize_video_data()