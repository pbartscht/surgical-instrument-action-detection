#!/usr/bin/env python3

"""
VOC to YOLO Converter for Surgical Instrument Detection Dataset
------------------------------------------------------------
Converts annotations from Jin et al. (2018) VOC format to YOLO format.

Usage:
    python voc_to_yolo_converter.py \
        --input-dir /path/to/m2cai16-tool-locations \
        --output-dir /path/to/yolo_dataset

See README.md in this directory for detailed information.
"""

import os
import xml.etree.ElementTree as ET
from glob import glob
import shutil
import logging
from pathlib import Path
import argparse
import yaml
from typing import Tuple, Optional
from datetime import datetime

class VOCtoYOLOConverter:
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the converter with input and output directories
        
        Args:
            input_dir: Path to the VOC dataset directory
            output_dir: Path where the YOLO dataset should be created
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.setup_logging()
        
        # Verify input directory structure
        self.annotations_dir = self.input_dir / 'Annotations'
        self.images_dir = self.input_dir / 'JPEGImages'
        self.imagesets_dir = self.input_dir / 'ImageSets' / 'Main'
        
        self.verify_input_structure()
        self.classes = self.load_classes()

    def setup_logging(self):
        """Setup logging to both file and console"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.output_dir / f'conversion_{timestamp}.log'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Starting conversion from {self.input_dir} to {self.output_dir}")

    def verify_input_structure(self):
        """Verify that all required input directories and files exist"""
        if not self.annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {self.annotations_dir}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.imagesets_dir.exists():
            raise FileNotFoundError(f"ImageSets directory not found: {self.imagesets_dir}")

    def load_classes(self) -> list:
        """Load class names from the class list file"""
        class_file = self.input_dir / 'class_list.txt'
        try:
            with open(class_file, 'r') as f:
                classes = [line.split()[1] for line in f.readlines()]
                logging.info(f"Loaded {len(classes)} classes: {classes}")
                return classes
        except FileNotFoundError:
            raise FileNotFoundError(f"Class list file not found: {class_file}")
        except IndexError:
            raise ValueError(f"Invalid format in class file: {class_file}")

    def convert_bbox(self, size: Tuple[int, int], box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """
        Convert VOC format bounding box to YOLO format
        
        Args:
            size: Image size (width, height)
            box: Bounding box in VOC format (xmin, xmax, ymin, ymax)
        
        Returns:
            tuple: Bounding box in YOLO format (x_center, y_center, width, height)
        """
        dw, dh = 1.0 / size[0], 1.0 / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        return (x * dw, y * dh, w * dw, h * dh)

    def convert_annotation(self, xml_path: Path) -> Optional[str]:
        """Convert a single XML annotation file to YOLO format"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            size = root.find('size')
            if size is None:
                raise ValueError("No size information found in XML")
                
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            filename = root.find('filename').text
            output_file = self.output_dir / 'labels' / f"{Path(filename).stem}.txt"

            with open(output_file, 'w') as out_file:
                for obj in root.iter('object'):
                    cls = obj.find('name').text
                    if cls not in self.classes:
                        logging.warning(f"Unknown class '{cls}' in {xml_path.name}")
                        continue

                    cls_id = self.classes.index(cls)
                    xmlbox = obj.find('bndbox')
                    b = (
                        float(xmlbox.find('xmin').text),
                        float(xmlbox.find('xmax').text),
                        float(xmlbox.find('ymin').text),
                        float(xmlbox.find('ymax').text)
                    )
                    bb = self.convert_bbox((w, h), b)
                    out_file.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")

            return filename

        except ET.ParseError as e:
            logging.error(f"Failed to parse {xml_path.name}: {e}")
        except Exception as e:
            logging.error(f"Error processing {xml_path.name}: {e}")
        return None

    def create_dataset_structure(self):
        """Create YOLO dataset directory structure"""
        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels').mkdir(parents=True, exist_ok=True)

    def process_splits(self):
        """Process train/val/test splits"""
        splits = ['train', 'val', 'trainval', 'test']
        
        for split in splits:
            split_file = self.imagesets_dir / f'{split}.txt'
            if not split_file.exists():
                logging.warning(f"Split file not found: {split}.txt")
                continue

            output_split = self.output_dir / f'{split}.txt'
            with open(split_file, 'r') as f, open(output_split, 'w') as out_f:
                filenames = [line.strip() + '.jpg' for line in f.readlines() if line.strip()]
                for filename in filenames:
                    image_path = self.output_dir / 'images' / filename
                    if image_path.exists():
                        out_f.write(f"{image_path}\n")
                    else:
                        logging.warning(f"Image not found for split {split}: {filename}")

    def create_yaml_config(self):
        """Create YOLO configuration YAML file"""
        config = {
            'train': str(self.output_dir / 'train.txt'),
            'val': str(self.output_dir / 'val.txt'),
            'test': str(self.output_dir / 'test.txt'),
            'nc': len(self.classes),
            'names': self.classes
        }
        
        with open(self.output_dir / 'data.yaml', 'w') as f:
            yaml.dump(config, f, sort_keys=False)

    def convert(self):
        """Run the complete conversion process"""
        logging.info("Starting conversion process...")
        self.create_dataset_structure()
        
        # Convert annotations
        xml_files = list(self.annotations_dir.glob('*.xml'))
        if not xml_files:
            raise FileNotFoundError(f"No XML files found in {self.annotations_dir}")

        logging.info(f"Found {len(xml_files)} XML files to convert")
        successful = 0
        
        for xml_file in xml_files:
            filename = self.convert_annotation(xml_file)
            if filename:
                src_image = self.images_dir / filename
                if src_image.exists():
                    shutil.copy(src_image, self.output_dir / 'images' / filename)
                    successful += 1
                else:
                    logging.warning(f"Image file not found: {src_image}")

        logging.info(f"Successfully converted {successful} out of {len(xml_files)} files")
        
        # Process splits and create config
        self.process_splits()
        self.create_yaml_config()
        
        logging.info(f"Conversion completed. Dataset prepared in: {self.output_dir}")
        logging.info(f"You can find the conversion log in: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Convert VOC format annotations to YOLO format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Path to the VOC dataset directory containing Annotations, JPEGImages, and ImageSets folders'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Path where the YOLO format dataset should be created'
    )

    args = parser.parse_args()
    
    try:
        converter = VOCtoYOLOConverter(args.input_dir, args.output_dir)
        converter.convert()
    except Exception as e:
        logging.error(f"Conversion failed: {e}")
        raise

if __name__ == '__main__':
    main()