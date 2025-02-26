"""
Surgical Verb Recognition Model with Instrument Constraints

This PyTorch Lightning module performs verb recognition in surgical videos by combining:
- Visual features from surgical tool-tissue interactions
- Known constraints between surgical instruments and possible actions

The model uses EfficientNet-B0 as backbone and supports 9 surgical verb classes 
across 6 different surgical instruments. Instrument-specific constraints are 
implemented using a soft masking approach during training and inference.

Key Features:
- Automatic masking of invalid instrument-verb combinations
- Comprehensive evaluation metrics per verb and instrument-verb pair
- Soft constraint implementation for stable training
- Custom loss function combining classification and constraint objectives
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import average_precision_score, precision_score, recall_score
import numpy as np
import timm
from typing import Dict, List, Tuple
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

class SurgicalVerbRecognition(pl.LightningModule):
    """
    PyTorch Lightning module for surgical verb recognition with instrument constraints.
    
    The model combines visual features from surgical videos with known constraints
    between instruments and valid actions to predict surgical verbs.
    
    Args:
        num_classes (int): Number of verb classes (default: 9)
        learning_rate (float): Learning rate for classifier (default: 5e-6)
        backbone_learning_rate (float): Learning rate for backbone (default: 5e-7)
        dropout (float): Dropout rate for classifier (default: 0.7)
        constraint_weight (float): Weight for constraint loss (default: 0.1)
        masking_value (float): Value for masking invalid actions (default: -10.0)
        
    Example:
        >>> model = SurgicalVerbRecognition()
        >>> output = model(images, instrument_names)
        >>> predictions = output['probabilities'].argmax(dim=1)
    """
    
    def __init__(
        self,
        num_classes: int = 9,
        learning_rate: float = 5e-6,
        backbone_learning_rate: float = 5e-7,
        dropout: float = 0.7,
        constraint_weight: float = 0.1,
        masking_value: float = -10.0
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Mapping of verb names to class indices
        self.verb_classes = {
            'dissect': 0,
            'retract': 1,
            'null_verb': 2,
            'coagulate': 3,
            'grasp': 4,
            'clip': 5,
            'aspirate': 6,
            'cut': 7,
            'irrigate': 8
        }
        
        # Mapping of instrument names to indices
        self.instrument_classes = {
            'grasper': 0,
            'bipolar': 1,
            'hook': 2,
            'scissors': 3,
            'clipper': 4,
            'irrigator': 5
        }
        
        # Valid instrument-verb combinations based on surgical domain knowledge
        self.VALID_PAIRS = {
            'grasper': ['grasp', 'retract', 'null_verb'],
            'hook': ['dissect', 'cut', 'null_verb', 'coagulate'],
            'bipolar': ['coagulate', 'dissect', 'null_verb'],
            'clipper': ['clip', 'null_verb'],
            'scissors': ['cut', 'null_verb'],
            'irrigator': ['aspirate', 'irrigate', 'null_verb']
        }
        
        # Create and register constraint matrix
        self.register_buffer('constraint_matrix', self._create_constraint_matrix())
        
        # Base classification loss
        self.criterion = nn.CrossEntropyLoss()
        
        self.backbone = timm.create_model(
            'mobilenetv3_small_100',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        #self.backbone = timm.create_model(
        #    'vit_small_patch16_224', 
        #    pretrained=True,
        #    num_classes=0,
        #    global_pool='avg'
        #)
        backbone_dim = 1024  

        # Classifier entsprechend anpassen
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

    def _create_constraint_matrix(self) -> torch.Tensor:
        """
        Creates binary matrix encoding valid instrument-verb combinations.
        
        Returns:
            torch.Tensor: Binary matrix of shape [num_instruments, num_verbs]
        """
        matrix = torch.zeros((len(self.instrument_classes), len(self.verb_classes)))
        
        for instrument, verbs in self.VALID_PAIRS.items():
            instrument_idx = self.instrument_classes[instrument]
            for verb in verbs:
                verb_idx = self.verb_classes[verb]
                matrix[instrument_idx, verb_idx] = 1.0
                
        return matrix

    def forward(self, x: torch.Tensor, instrument_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass applying instrument-specific constraints.
        
        Args:
            x: Batch of images [B, C, H, W]
            instrument_names: List of instrument names for the batch
            
        Returns:
            Dict containing:
                - logits: Raw model outputs after constraint masking
                - probabilities: Softmaxed probabilities
        """
        # Extract visual features
        visual_features = self.backbone(x)
        #print(f"Visual features shape: {visual_features.shape}")

        # Get verb logits
        logits = self.classifier(visual_features)

        # Remove the last dimension (pack class)
        logits = logits[:, :9]  # Quick fix: only use first 9 classes
        
        # Apply instrument-specific constraints
        instrument_indices = torch.tensor([
            self.instrument_classes[name] for name in instrument_names
        ], device=self.device)
        
        batch_constraints = self.constraint_matrix[instrument_indices]
        
        # Soft masking of invalid actions
        masked_logits = logits * batch_constraints + \
                       (1 - batch_constraints) * self.hparams.masking_value
        
        # Calculate probabilities
        probs = torch.softmax(masked_logits, dim=-1)
        
        return {
            'logits': masked_logits,
            'probabilities': probs
        }

    def training_step(
        self, 
        batch: Tuple[torch.Tensor, List[str], torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Training step combining classification and constraint losses.
        
        Args:
            batch: Tuple of (images, instrument_names, verb_labels)
            batch_idx: Index of current batch
            
        Returns:
            Total loss combining classification and constraint terms
        """
        images, instrument_names, labels = batch
        outputs = self(images, instrument_names)
        
        # Classification loss
        classification_loss = self.criterion(outputs['logits'], labels)
        
        # Constraint violation loss
        instrument_indices = torch.tensor([
            self.instrument_classes[name] for name in instrument_names
        ], device=self.device)
        batch_constraints = self.constraint_matrix[instrument_indices]
        constraint_loss = (outputs['probabilities'] * (1 - batch_constraints)).mean()
        
        # Combined loss
        total_loss = classification_loss + \
                    self.hparams.constraint_weight * constraint_loss
        
        # Logging
        self.log('train/classification_loss', classification_loss, prog_bar=True)
        self.log('train/constraint_loss', constraint_loss, prog_bar=True)
        self.log('train/total_loss', total_loss, prog_bar=True)
        
        return total_loss

    def validation_step(
        self, 
        batch: Tuple[torch.Tensor, List[str], torch.Tensor], 
        batch_idx: int
    ) -> None:
        """
        Validation step collecting metrics for epoch end computation.
        
        Args:
            batch: Tuple of (images, instrument_names, verb_labels)
            batch_idx: Index of current batch
        """
        images, instrument_names, labels = batch
        outputs = self(images, instrument_names)
        val_loss = self.criterion(outputs['logits'], labels)
        
        # Store outputs for epoch_end metrics
        instrument_indices = torch.tensor([
            self.instrument_classes[name] for name in instrument_names
        ], device=self.device)
        
        self.validation_step_outputs.append({
            'val_loss': val_loss,
            'probs': outputs['probabilities'].detach(),
            'labels': labels.detach(),
            'instrument_indices': instrument_indices.detach()
        })
        
        self.log('val/loss', val_loss, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        Computes comprehensive validation metrics including:
        - Per instrument-verb pair metrics (AP, Precision, Recall)
        - Mean metrics across all valid combinations
        - Weighted global metrics
        """
        # Collect all outputs
        all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        all_instruments = torch.cat([x['instrument_indices'] for x in self.validation_step_outputs])
        all_preds = torch.argmax(all_probs, dim=1)
        
        # Per instrument-verb pair metrics
        metrics = {'ap': [], 'precision': [], 'recall': []}
        
        for instrument_name, instrument_idx in self.instrument_classes.items():
            # Get mask for current instrument
            instrument_mask = (all_instruments == instrument_idx)
            instrument_samples = instrument_mask.sum().item()
            
            if instrument_samples > 0:
                # Get probabilities and predictions for current instrument
                inst_probs = all_probs[instrument_mask]
                inst_labels = all_labels[instrument_mask]
                inst_preds = all_preds[instrument_mask]
                
                # Calculate metrics only for valid verbs for this instrument
                valid_verbs = self.VALID_PAIRS[instrument_name]
                for verb in valid_verbs:
                    verb_idx = self.verb_classes[verb]
                    
                    # Create binary labels for current verb
                    binary_labels = (inst_labels == verb_idx).cpu().numpy()
                    verb_probs = inst_probs[:, verb_idx].cpu().numpy()
                    binary_preds = (inst_preds == verb_idx).cpu().numpy()
                    
                    # Only calculate metrics if we have positive samples
                    if binary_labels.sum() > 0:
                        ap = average_precision_score(binary_labels, verb_probs)
                        prec = precision_score(binary_labels, binary_preds, zero_division=0)
                        rec = recall_score(binary_labels, binary_preds, zero_division=0)
                        
                        metrics['ap'].append(ap)
                        metrics['precision'].append(prec)
                        metrics['recall'].append(rec)
                        
                        # Log per-pair metrics
                        pair_name = f"{instrument_name}-{verb}"
                        self.log(f'val/AP_{pair_name}', ap, prog_bar=False)
                        self.log(f'val/precision_{pair_name}', prec, prog_bar=False)
                        self.log(f'val/recall_{pair_name}', rec, prog_bar=False)
        
        # Log average metrics
        if len(metrics['ap']) > 0:  # Only log if we have valid metrics
            self.log('val/mAP', np.mean(metrics['ap']), prog_bar=True)
            self.log('val/mean_precision', np.mean(metrics['precision']), prog_bar=True)
            self.log('val/mean_recall', np.mean(metrics['recall']), prog_bar=True)
        
        # Log weighted global metrics
        self.log('val/weighted_precision', precision_score(
            all_labels.cpu(),
            all_preds.cpu(),
            average='weighted',
            zero_division=0
        ), prog_bar=True)
        
        self.log('val/weighted_recall', recall_score(
            all_labels.cpu(),
            all_preds.cpu(),
            average='weighted',
            zero_division=0
        ), prog_bar=True)
        
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