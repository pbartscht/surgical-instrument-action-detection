import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import precision_score, recall_score
import timm
from typing import Dict, List, Tuple
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

class UnconstrainedSurgicalVerbRecognition(pl.LightningModule):
    """
    PyTorch Lightning module for surgical verb recognition without instrument constraints.
    
    This version removes the instrument-specific constraints and treats the problem
    as a pure classification task based only on visual features.
    
    Args:
        num_classes (int): Number of verb classes (default: 9)
        learning_rate (float): Learning rate for classifier (default: 5e-6)
        backbone_learning_rate (float): Learning rate for backbone (default: 5e-7)
        dropout (float): Dropout rate for classifier (default: 0.7)
    """
    
    def __init__(
        self,
        num_classes: int = 9,
        learning_rate: float = 5e-6,
        backbone_learning_rate: float = 5e-7,
        dropout: float = 0.7
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
        
        # Base classification loss
        self.criterion = nn.CrossEntropyLoss()
        
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
        Forward pass computing verb predictions from images.
        
        Args:
            x: Batch of images [B, C, H, W]
            
        Returns:
            Dict containing:
                - logits: Raw model outputs
                - probabilities: Softmaxed probabilities
        """
        # Extract visual features
        visual_features = self.backbone(x)
        
        # Get verb logits
        logits = self.classifier(visual_features)
        
        # Remove the last dimension (pack class)
        logits = logits[:, :9]  # Quick fix: only use first 9 classes
        
        # Calculate probabilities
        probs = torch.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'probabilities': probs
        }

    def training_step(
        self, 
        batch: Tuple[torch.Tensor, List[str], torch.Tensor], 
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
        images, _, labels = batch  # Instrument names are ignored
        outputs = self(images)
        
        # Classification loss
        loss = self.criterion(outputs['logits'], labels)
        
        # Logging
        self.log('train/loss', loss, prog_bar=True)
        
        return loss

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
        images, _, labels = batch  # Instrument names are ignored
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