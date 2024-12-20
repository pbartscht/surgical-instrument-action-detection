#!/usr/bin/env python3

from pathlib import Path
import sys
import yaml
from utils.custom_yolo import CustomYOLO

def test_augmentations():
    # Add project root to Python path
    project_root = Path(__file__).parent.parent.absolute()
    sys.path.append(str(project_root))
    
    # Define paths
    pretrained_weights_path = project_root / 'weights' / 'pretrained' / 'yolo11l.pt'
    
    print("\n=== Testing Augmentations ===")
    try:
        # Initialize model
        model = CustomYOLO(str(pretrained_weights_path))
        
        # Load dataset config
        data_yaml_path = str(project_root / 'config' / 'model_config' / 'data.yaml')
        print(f"Loading dataset from: {data_yaml_path}")
        
        if not Path(data_yaml_path).exists():
            raise FileNotFoundError(f"Could not find data.yaml at: {data_yaml_path}")
            
        # Verify yaml can be loaded
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            
        # Create dataset using get_dataset method from CustomYOLO
        dataset = model.get_dataset(
            dataset_path=data_yaml_path,
            mode='train'
        )
        
        # Test multiple images
        for i in range(3):  # Test first 3 images
            print(f"\nTesting image {i+1}:")
            img, labels = dataset[i]
            print(f"Image shape: {img.shape}")
            print(f"Number of labels: {len(labels) if labels is not None else 0}")
            if labels is not None:
                print(f"Label format: {labels}")
        
        print("\nAugmentation settings:")
        print(f"Augmentations active: {dataset.augment}")
        if hasattr(dataset, 'instrument_aug'):
            print(f"Surgical augmentations initialized: {dataset.instrument_aug is not None}")
        
    except Exception as e:
        print(f"Error while testing augmentations: {str(e)}")
        import traceback
        traceback.print_exc()
    print("\n=== Augmentation Test Complete ===")

if __name__ == '__main__':
    test_augmentations()