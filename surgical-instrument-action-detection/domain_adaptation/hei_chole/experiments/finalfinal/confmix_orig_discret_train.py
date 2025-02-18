from pathlib import Path
import yaml
import os
from ultralytics import YOLO
import torch
import torch.nn as nn

class ConfMixLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        
    def forward(self, predictions, targets, mixed_predictions=None, mixed_targets=None):
        # Standard detection loss
        det_loss = self.compute_detection_loss(predictions, targets)
        
        # Consistency loss if we have mixed samples
        if mixed_predictions is not None and mixed_targets is not None:
            consistency_loss = self.compute_consistency_loss(mixed_predictions, mixed_targets)
            # Dynamic weighting based on confidence scores
            gamma = self.compute_confidence_weight(mixed_predictions)
            return det_loss + gamma * consistency_loss
        
        return det_loss
    
    def compute_detection_loss(self, predictions, targets):
        """Standard YOLOv5 detection loss"""
        cls_loss = self.bce(predictions['cls'], targets['cls'])
        box_loss = self.l1(predictions['box'], targets['box'])
        obj_loss = self.bce(predictions['obj'], targets['obj'])
        return cls_loss + box_loss + obj_loss
    
    def compute_consistency_loss(self, predictions, targets):
        """Consistency loss between mixed predictions and targets"""
        # Similar to detection loss but for mixed samples
        cls_loss = self.bce(predictions['cls'], targets['cls'])
        box_loss = self.l1(predictions['box'], targets['box'])
        return cls_loss + box_loss
    
    def compute_confidence_weight(self, predictions):
        """Dynamic weighting based on prediction confidence"""
        conf_scores = torch.sigmoid(predictions['obj'])
        # Count predictions above threshold (e.g., 0.5)
        reliable_preds = (conf_scores > 0.5).float().sum()
        total_preds = conf_scores.numel()
        # Weight based on ratio of reliable predictions
        gamma = reliable_preds / (total_preds + 1e-6)
        return gamma.clamp(0.1, 0.9)  # Limit range to [0.1, 0.9]

def load_dataset_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded dataset config from {yaml_path}")
    return config

def setup_training(pretrained_model_path, dataset_yaml_path, project_name="confmix_training_strategy"):
    # Load dataset configuration
    dataset_config = load_dataset_config(dataset_yaml_path)
    
    # Load pretrained model
    model = YOLO(pretrained_model_path)
    
    # Add custom loss
    model.loss_fn = ConfMixLoss()
    
    # Training arguments
    training_args = {
        'data': dataset_yaml_path,
        'imgsz': 640,
        'epochs': 35,
        'batch': 16,
        'patience': 10,
        
        # Learning rate settings
        'lr0': 0.001,
        'lrf': 0.01,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        
        # Optimizer settings
        'optimizer': 'AdamW',
        'weight_decay': 0.001,
        'momentum': 0.937,
        
        # Memory optimization
        'cache': False,
        'workers': 4,
        'overlap_mask': False,
        
        # Regularization
        'dropout': 0.1,
        'label_smoothing': 0.1,
        
        # Project settings
        'project': project_name,
        'name': 'confmix_training',
        'exist_ok': True,
        'pretrained': True,
        'save_period': 5,
        
        # Validation
        'val': True,
        'save': True,
    }
    
    try:
        results = model.train(**training_args)
        return results
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def main():
    PRETRAINED_MODEL = "/home/Bartscht/YOLO/surgical-instrument-action-detection/domain_adaptation/hei_chole/experiments/finalfinal/heichole_transfer_balanced_instruments/transfer_learning/weights/last.pt"
    DATASET_YAML = "/data/Bartscht/balanced_mixed_samples_epoch1/dataset.yaml"
    
    print("Starting ConfMix training...")
    print(f"Using pretrained model: {PRETRAINED_MODEL}")
    print(f"Using dataset config: {DATASET_YAML}")
    
    try:
        results = setup_training(
            PRETRAINED_MODEL,
            DATASET_YAML,
            project_name="transfer_epoch1"
        )
        
        print("\nTraining completed successfully!")
        print("Results summary:")
        print(f"Best mAP: {results.maps}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")

if __name__ == "__main__":
    main()