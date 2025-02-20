from pathlib import Path
import yaml
import os
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaussianYOLOLoss(nn.Module):
    """
    Implementiert die Gaussian-basierte Loss-Funktion für YOLO Bounding Boxes
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_bbox, target_bbox):
        # Extrahiere Mean und Variance aus den Predictions
        # pred_bbox: [batch_size, 8] (µx, µy, µw, µh, σx, σy, σw, σh)
        pred_mean = pred_bbox[..., :4]
        pred_var = torch.exp(pred_bbox[..., 4:])  # Variance muss positiv sein
        
        # Berechne Negative Log Likelihood
        nll = 0.5 * torch.log(2 * math.pi * pred_var)
        nll += 0.5 * ((target_bbox - pred_mean) ** 2) / pred_var
        
        return nll.mean()

class FocalLoss(nn.Module):
    """
    Implementiert Focal Loss für bessere Behandlung von Klassenimbalancen
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, predictions, targets):
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        p = torch.sigmoid(predictions)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        return loss.mean()

class ImprovedConfMixLoss(nn.Module):
    def __init__(self, num_epochs):
        super().__init__()
        self.gaussian_loss = GaussianYOLOLoss()
        self.focal_loss = FocalLoss()
        self.num_epochs = num_epochs
        
    def forward(self, predictions, targets, mixed_predictions=None, mixed_targets=None, epoch=0):
        # Standard detection loss
        det_loss = self.compute_detection_loss(predictions, targets)
        
        # Consistency loss wenn mixed samples vorhanden
        if mixed_predictions is not None and mixed_targets is not None:
            consistency_loss = self.compute_consistency_loss(
                mixed_predictions, 
                mixed_targets,
                epoch
            )
            # Dynamische Gewichtung basierend auf Confidence und Trainingsfortschritt
            gamma = self.compute_adaptive_weight(mixed_predictions, epoch)
            return det_loss + gamma * consistency_loss
        
        return det_loss
    
    def compute_detection_loss(self, predictions, targets):
        """Verbesserter Detection Loss mit Gaussian und Focal Loss"""
        cls_loss = self.focal_loss(predictions['cls'], targets['cls'])
        box_loss = self.gaussian_loss(predictions['box'], targets['box'])
        obj_loss = self.focal_loss(predictions['obj'], targets['obj'])
        
        # Gewichtung der verschiedenen Loss-Komponenten
        return cls_loss + 2.0 * box_loss + obj_loss
    
    def compute_consistency_loss(self, predictions, targets, epoch):
        """Verbesserter Consistency Loss mit progressiver Anpassung"""
        # Berechne Confidence-basierte Gewichte
        confidence_weights = self.compute_confidence_weight(predictions, epoch)
        
        # Gewichtete Loss-Berechnung
        cls_loss = self.focal_loss(predictions['cls'], targets['cls']) * confidence_weights
        box_loss = self.gaussian_loss(predictions['box'], targets['box']) * confidence_weights
        
        return cls_loss + box_loss
    
    def compute_confidence_weight(self, predictions, epoch):
        """Berechnet Confidence-Gewichte mit Box-Uncertainty"""
        base_conf = torch.sigmoid(predictions['obj'])
        
        # Box uncertainty aus Gaussian parameters
        box_params = predictions['box']
        box_variance = torch.exp(box_params[..., 4:])  # σ²
        box_uncertainty = 1.0 - torch.mean(1.0 / (1.0 + box_variance), dim=-1)
        
        # Progressive mixing
        progress = epoch / self.num_epochs
        alpha = 5.0
        delta = 2.0 / (1.0 + torch.exp(-alpha * progress)) - 1.0
        
        # Kombiniere base confidence mit box uncertainty
        combined_conf = (1.0 - delta) * base_conf + delta * (1.0 - box_uncertainty)
        
        return combined_conf

    def compute_adaptive_weight(self, predictions, epoch):
        """Berechnet adaptives Gewicht für Consistency Loss"""
        conf_scores = self.compute_confidence_weight(predictions, epoch)
        
        # Anzahl verlässlicher Predictions (conf > 0.5)
        reliable_preds = (conf_scores > 0.5).float().sum()
        total_preds = conf_scores.numel()
        
        # Basis-gamma basierend auf verlässlichen Predictions
        base_gamma = reliable_preds / (total_preds + 1e-6)
        
        # Progressive Anpassung
        progress = epoch / self.num_epochs
        gamma = base_gamma * (1.0 - math.exp(-3.0 * progress))
        
        return gamma.clamp(0.1, 0.9)

def setup_training(pretrained_model_path, dataset_yaml_path, project_name="confmix_last"):
    # Lade Dataset Konfiguration
    with open(dataset_yaml_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Lade vortrainiertes Modell
    model = YOLO(pretrained_model_path)
    
    # Training Parameter
    epochs = 30  # Anzahl der Epochs
    
    # Initialisiere verbesserten Loss
    model.loss_fn = ImprovedConfMixLoss(num_epochs=epochs)
    
    # Training Argumente
    training_args = {
        'data': dataset_yaml_path,
        'imgsz': 640,
        'epochs': epochs,
        'batch': 8,
        'patience': 50,  # Erhöhte Patience für bessere Konvergenz
        
        # Learning rate settings mit Cosine Annealing
        'lr0': 0.001,
        'lrf': 0.01,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        
        # Optimizer settings
        'optimizer': 'AdamW',  # Verwende AdamW statt SGD
        'weight_decay': 0.05,  # Erhöhtes weight decay
        'momentum': 0.937,
        
        # Augmentation und Regularisierung
        'hsv_h': 0.015,  # Hue augmentation
        'hsv_s': 0.7,    # Saturation augmentation
        'hsv_v': 0.4,    # Value augmentation
        'degrees': 0.0,   # Rotation
        'translate': 0.1, # Translation
        'scale': 0.5,    # Scaling
        'shear': 0.0,    # Shear
        'perspective': 0.0, # Perspective
        'flipud': 0.0,   # Vertical flip
        'fliplr': 0.5,   # Horizontal flip
        'mosaic': 1.0,   # Mosaic augmentation
        'mixup': 0.0,    # Mixup augmentation
        
        # Dropout und Regularisierung
        'dropout': 0.2,  # Erhöhter Dropout
        'label_smoothing': 0.1,
        
        # Project settings
        'project': project_name,
        'name': 'confmix_improved_training',
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
    PRETRAINED_MODEL = "/data/Bartscht/YOLO/best_v35.pt"
    DATASET_YAML = "/data/Bartscht/balanced_mixed_samples_epoch_last/dataset.yaml"
    
    print("Starting improved ConfMix training...")
    print(f"Using pretrained model: {PRETRAINED_MODEL}")
    print(f"Using dataset config: {DATASET_YAML}")
    
    try:
        results = setup_training(
            PRETRAINED_MODEL,
            DATASET_YAML,
            project_name="correct_transfer"
        )
        
        print("\nTraining completed successfully!")
        print("Results summary:")
        print(f"Best mAP: {results.maps}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")

if __name__ == "__main__":
    main()