#!/usr/bin/env python3
"""
YOLO Surgical Instrument Class Weight Updater
===========================================

This script analyzes the class distribution in a YOLO format dataset specifically for
surgical instrument detection and updates the class weights in the data.yaml file.
It uses relative paths based on the project structure.

Features:
- Uses relative paths for project navigation
- Analyzes current class distribution in training set
- Calculates balanced class weights
- Updates data.yaml with new class weights
- Supports special handling of ignored classes (e.g., SpecimenBag)
"""

import os
import yaml
import logging
from collections import Counter
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SurgicalInstrumentWeightUpdater:
    def __init__(self):
        """Initialize with relative paths based on script location"""
        # Get the script's directory (utils)
        self.script_dir = Path(__file__).parent.absolute()
        
        # Navigate to data.yaml location (../../model_config/data.yaml)
        self.yaml_path = (self.script_dir / '..' / '..' / 'model_config' / 'data.yaml').resolve()
        
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at {self.yaml_path}")
            
        self.config = self._load_yaml()
        self.base_path = Path(self.config['path'])
        
    def _load_yaml(self) -> dict:
        """Load and validate data.yaml configuration"""
        with open(self.yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            
        required_keys = ['path', 'train', 'names', 'nc']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required keys in data.yaml: {missing_keys}")
            
        return config
    
    def analyze_class_distribution(self) -> Counter:
        """Analyze class distribution in training set"""
        train_labels_path = self.base_path / 'labels' / 'train'
        class_counts = Counter()
        
        if not train_labels_path.exists():
            raise FileNotFoundError(f"Training labels directory not found at {train_labels_path}")
        
        for label_file in train_labels_path.glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
            except Exception as e:
                logger.error(f"Error processing {label_file}: {e}")
                
        return class_counts
    
    def calculate_class_weights(self, class_counts: Counter) -> dict:
        """Calculate balanced class weights using inverse frequency"""
        total_samples = sum(class_counts.values())
        n_classes = len(self.config['names'])
        weights = {}
        
        for class_id in range(n_classes):
            count = class_counts[class_id]
            # Handle SpecimenBag (class 6) specially
            if class_id == 6:  # SpecimenBag
                weights[class_id] = 0.0  # Keep ignored
            elif count > 0:
                # Calculate inverse frequency weight
                weights[class_id] = round(total_samples / (n_classes * count), 2)
            else:
                weights[class_id] = 0.0
                logger.warning(f"Class {self.config['names'][class_id]} has no samples!")
                
        return weights
    
    def update_yaml(self, class_weights: dict):
        """Update data.yaml with new class weights"""
        self.config['class_weights'] = class_weights
        
        # Create backup of original yaml
        backup_path = self.yaml_path.with_suffix('.yaml.backup')
        if not backup_path.exists():
            import shutil
            shutil.copy2(self.yaml_path, backup_path)
            logger.info(f"Created backup at {backup_path}")
        
        with open(self.yaml_path, 'w') as f:
            yaml.dump(self.config, f, sort_keys=False)
            
        logger.info(f"Updated class weights in {self.yaml_path}")
        
    def run(self):
        """Run the complete weight update process"""
        logger.info(f"Using data.yaml at: {self.yaml_path}")
        logger.info(f"Base dataset path: {self.base_path}")
        logger.info("\nAnalyzing class distribution...")
        
        class_counts = self.analyze_class_distribution()
        
        logger.info("\nCurrent class distribution:")
        for class_id, count in class_counts.items():
            class_name = self.config['names'][class_id]
            logger.info(f"{class_name}: {count} samples")
        
        weights = self.calculate_class_weights(class_counts)
        
        logger.info("\nCalculated class weights:")
        for class_id, weight in weights.items():
            class_name = self.config['names'][class_id]
            logger.info(f"{class_name}: {weight}")
            
        self.update_yaml(weights)

def main():
    """Main execution"""
    try:
        updater = SurgicalInstrumentWeightUpdater()
        updater.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()