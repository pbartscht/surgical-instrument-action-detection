import os
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path
import sys
from augm_dataloader import BasicSurgicalYOLODataset

class CustomYOLO(YOLO):
    """
    Custom YOLO implementation with support for class weights and surgical dataset.
    Extends the base YOLO class from ultralytics to add class-specific weight handling
    for improved training on imbalanced datasets.
    """
    
    def __init__(self, model):
        """
        Initialize CustomYOLO with model and load class weights from config.
        
        Args:
            model: Path to YOLO model or model name
        """
        super().__init__(model)
        
        # Load class weights from config/model_config/data.yaml
        config_path = Path(__file__).parents[2] / 'config' / 'model_config' / 'data.yaml'
        try:
            with open(config_path, 'r') as file:
                data_config = yaml.safe_load(file)
                # Convert class weights dict to tensor
                weights_dict = data_config.get('class_weights', {})
                self.num_classes = data_config.get('nc', 7)
                self.class_weights = torch.ones(self.num_classes)
                
                for idx, weight in weights_dict.items():
                    self.class_weights[int(idx)] = weight
                
                # Move weights to appropriate device
                self.class_weights = self.class_weights.to('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Loaded class weights: {self.class_weights}")
                
        except Exception as e:
            print(f"Warning: Could not load class weights: {e}")
            self.class_weights = None

    def get_dataset(self, dataset_path, mode='train', batch=None):
        """
        Create and return a BasicSurgicalYOLODataset instance.
        """
        print(f"Erstelle Dataset für Modus: {mode}")
        print(f"Verwende Konfiguration aus: {dataset_path}")
        
        return BasicSurgicalYOLODataset(
            dataset_path,  # Übergebe den YAML-Pfad
            augment=(mode == 'train'),
            class_weights=self.class_weights
        )

    def _apply_class_weights(self, cls_loss, cls_targets):
        """
        Apply class-specific weights to classification loss.
        
        Args:
            cls_loss (torch.Tensor): Classification loss
            cls_targets (torch.Tensor): Target class indices
        
        Returns:
            torch.Tensor: Weighted classification loss
        """
        if self.class_weights is not None and cls_targets is not None:
            weights = self.class_weights[cls_targets.long()]
            return cls_loss * weights
        return cls_loss

    def criterion(self, preds, targets):
        """
        Override default loss calculation to include class weights.
        
        Args:
            preds (dict): Model predictions
            targets (dict): Ground truth targets
        
        Returns:
            ComputeLoss: Modified loss including class weights
        """
        loss = super().criterion(preds, targets)
        
        # Modify only the classification loss component
        if hasattr(loss, 'cls') and self.class_weights is not None:
            cls_targets = targets.get('cls')
            if cls_targets is not None:
                loss.cls = self._apply_class_weights(loss.cls, cls_targets)
        
        return loss