import os
import yaml
from ultralytics import YOLO
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from augm_dataloader import BasicSurgicalYOLODataset

class CustomYOLO(YOLO):
    def __init__(self, model):
        super().__init__(model)
        
        # Verwende relativen Pfad zur config/data.yaml
        config_path = Path(__file__).parents[2] / 'config' / 'data.yaml'
        try:
            with open(config_path, 'r') as file:
                data_config = yaml.safe_load(file)
                self.class_weights = data_config.get('class_weights', None)
        except Exception as e:
            print(f"Warning: Could not load class weights: {e}")
            self.class_weights = None

    def get_dataset(self, dataset_path, mode='train', batch=None):
        return BasicSurgicalYOLODataset(
            dataset_path, 
            mode=mode,
            batch=batch,
            class_weights=self.class_weights
        )