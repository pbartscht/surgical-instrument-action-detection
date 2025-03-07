import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import average_precision_score, precision_score, recall_score
import numpy as np
import timm
from typing import Dict, List, Tuple
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

class SurgicalActionRecognition(pl.LightningModule):
    """
    PyTorch Lightning module for combined surgical instrument-verb pair recognition.
    
    The model directly classifies input images into one of 25 instrument-verb pairs
    using EfficientNet-B0 as backbone.
    
    Args:
        num_classes (int): Number of instrument-verb pair classes (default: 25)
        learning_rate (float): Learning rate for classifier (default: 5e-6)
        backbone_learning_rate (float): Learning rate for backbone (default: 5e-7)
        dropout (float): Dropout rate for classifier (default: 0.7)
        class_weights (torch.Tensor, optional): Custom weights for class imbalance
    
    Example:
        >>> model = SurgicalActionRecognition()
        >>> output = model(images)
        >>> predictions = output['probabilities'].argmax(dim=1)
    """
    
    def __init__(
        self,
        num_classes: int = 25,
        learning_rate: float = 5e-6,
        backbone_learning_rate: float = 5e-7,
        dropout: float = 0.7,
        class_weights: torch.Tensor = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Mapping of instrument-verb pairs to class indices (sorted by frequency)
        self.action_classes = {
            'Scissors-Coagulate': 0,
            'Grasper-Pack': 1,
            'Grasper-Dissect': 2,
            'Bipolar-Grasp': 3,
            'Scissors-Dissect': 4,
            'Scissors_null-Verb': 5,
            'Irrigator-Dissect': 6,
            'Bipolar-Retract': 7,
            'Clipper_null-Verb': 8,
            'Irrigator-Retract': 9,
            'Irrigator-Irrigate': 10,
            'Bipolar_null-Verb': 11,
            'Irrigator_null-Verb': 12,
            'Hook-Retract': 13,
            'Hook-Coagulate': 14,
            'Bipolar-Dissect': 15,
            'Grasper_null-Verb': 16,
            'Scissors-Cut': 17,
            'Irrigator-Aspirate': 18,
            'Hook_null-Verb': 19,
            'Clipper-Clip': 20,
            'Grasper-Grasp': 21,
            'Bipolar-Coagulate': 22,
            'Grasper-Retract': 23,
            'Hook-Dissect': 24
        }
        
        # Reverse mapping for evaluation
        self.idx_to_action = {v: k for k, v in self.action_classes.items()}
        
        # Setup class weights for loss function
        if class_weights is None:
            # Default weights based on provided frequencies
            default_weights = torch.tensor([
                566.92, 425.19, 103.08, 41.48, 40.49, 37.79, 28.35, 23.14, 14.85, 13.55,
                13.55, 13.08, 12.46, 7.70, 6.41, 4.63, 3.55, 2.75, 2.38, 1.39,
                1.38, 1.25, 1.04, 0.13, 0.09
            ], dtype=torch.float32)
            
            # Convert frequencies to weights (inverse frequency)
            class_weights = 1.0 / default_weights
            # Normalize weights
            class_weights = class_weights / class_weights.sum() * len(default_weights)
        
        self.register_buffer('class_weights', class_weights)
        
        # Base classification loss with class weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # EfficientNet-B0 backbone
        self.backbone = timm.create_model(
            'tf_efficientnet_b0_ns',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        backbone_dim = 1280
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Storage for validation metrics
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing instrument-verb pair predictions.
        
        Args:
            x: Batch of images [B, C, H, W]
            
        Returns:
            Dict containing:
                - logits: Raw model outputs
                - probabilities: Softmaxed probabilities
        """
        # Extract visual features
        visual_features = self.backbone(x)
        
        # Get logits
        logits = self.classifier(visual_features)
        
        # Calculate probabilities
        probs = torch.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'probabilities': probs
        }

    def training_step(
        self, 
        batch: Tuple[torch.Tensor, str, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Training step computing classification loss.
        
        Args:
            batch: Tuple of (images, instrument_names, verb_labels)
            batch_idx: Index of current batch
            
        Returns:
            Classification loss
        """
        images, instrument_names, labels = batch
        outputs = self(images)
        
        # Classification loss
        loss = self.criterion(outputs['logits'], labels)
        
        # Logging
        self.log('train/loss', loss, prog_bar=True)
        
        return loss

    def validation_step(
        self, 
        batch: Tuple[torch.Tensor, str, torch.Tensor], 
        batch_idx: int
    ) -> None:
        """
        Validation step collecting metrics for epoch end computation.
        
        Args:
            batch: Tuple of (images, instrument_names, verb_labels)
            batch_idx: Index of current batch
        """
        images, instrument_names, labels = batch
        outputs = self(images)
        val_loss = self.criterion(outputs['logits'], labels)
        
        # Store outputs for epoch_end metrics
        self.validation_step_outputs.append({
            'val_loss': val_loss,
            'probs': outputs['probabilities'].detach(),
            'labels': labels.detach()
        })
        
        self.log('val/loss', val_loss, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        Computes validation metrics including precision and recall.
        """
        # Collect all outputs
        all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        all_preds = torch.argmax(all_probs, dim=1)
        
        # Calculate overall metrics
        accuracy = (all_preds == all_labels).float().mean().item()
        
        # Weighted metrics to handle class imbalance
        weighted_precision = precision_score(
            all_labels.cpu(),
            all_preds.cpu(),
            average='weighted',
            zero_division=0
        )
        
        weighted_recall = recall_score(
            all_labels.cpu(),
            all_preds.cpu(),
            average='weighted',
            zero_division=0
        )
        
        # Log metrics
        self.log('val/accuracy', accuracy, prog_bar=True)
        self.log('val/weighted_precision', weighted_precision, prog_bar=True)
        self.log('val/weighted_recall', weighted_recall, prog_bar=True)
        
        # Per-class metrics for the most frequent and least frequent classes
        for action_name, action_idx in self.action_classes.items():
            # Create binary labels for current action
            binary_labels = (all_labels == action_idx).cpu().numpy()
            binary_preds = (all_preds == action_idx).cpu().numpy()
            action_probs = all_probs[:, action_idx].cpu().numpy()
            
            # Only calculate metrics if we have positive samples
            if binary_labels.sum() > 0:
                ap = average_precision_score(binary_labels, action_probs)
                prec = precision_score(binary_labels, binary_preds, zero_division=0)
                rec = recall_score(binary_labels, binary_preds, zero_division=0)
                
                # Log per-action metrics
                self.log(f'val/AP_{action_name}', ap, prog_bar=False)
                self.log(f'val/precision_{action_name}', prec, prog_bar=False)
                self.log(f'val/recall_{action_name}', rec, prog_bar=False)
        
        # Clear stored outputs
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """
        Configures optimizers and learning rate schedules.
        
        Returns:
            Dict containing optimizer and lr_scheduler configuration
        """
        # Different learning rates for backbone and classifier
        optimizer = torch.optim.AdamW([
            {'params': self.backbone.parameters(), 
             'lr': self.hparams.backbone_learning_rate},
            {'params': self.classifier.parameters(), 
             'lr': self.hparams.learning_rate}
        ])
        
        # Linear warm-up + cosine decay schedule
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1,
            total_iters=5
        )
        
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=40,
            eta_min=1e-7
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[5]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss"
            }
        }
        
    def extract_instrument_verb(self, action_idx):
        """
        Extracts instrument and verb from action index.
        
        Args:
            action_idx: Index of the action class
            
        Returns:
            Tuple of (instrument, verb)
        """
        action_name = self.idx_to_action[action_idx]
        parts = action_name.split('-')
        instrument = parts[0]
        verb = parts[1]
        return instrument, verb

    def predict_crop(self, image_tensor):
        """
        Make prediction for a single crop image.
        
        Args:
            image_tensor: Preprocessed image tensor [C, H, W]
            
        Returns:
            Dict with prediction results
        """
        # Ensure model is in eval mode
        self.eval()
        
        # Add batch dimension if needed
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            outputs = self(image_tensor)
        
        # Get prediction
        probs = outputs['probabilities'][0]  # Remove batch dimension
        action_idx = torch.argmax(probs).item()
        confidence = probs[action_idx].item()
        
        # Get action name and split into instrument and verb
        action_name = self.idx_to_action[action_idx]
        instrument, verb = self.extract_instrument_verb(action_idx)
        
        return {
            'action_idx': action_idx,
            'action_name': action_name,
            'instrument': instrument,
            'verb': verb,
            'confidence': confidence,
            'all_probs': probs.cpu().numpy()
        }