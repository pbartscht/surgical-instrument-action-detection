import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
import numpy as np
import timm
from typing import Dict, List, Tuple
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR


class SimpleVerbRecognition(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 10,
        learning_rate: float = 5e-5,
        backbone_learning_rate: float = 5e-6,
        dropout: float = 0.5,
        warmup_epochs: int = 5
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.verb_classes = {
            'grasp': 0, 'retract': 1, 'dissect': 2, 'coagulate': 3,
            'clip': 4, 'cut': 5, 'aspirate': 6, 'irrigate': 7,
            'pack': 8, 'null_verb': 9
        }
        
        self.instrument_classes = {
            'grasper': 0, 'bipolar': 1, 'hook': 2,
            'scissors': 3, 'clipper': 4, 'irrigator': 5
        }
        
        # Normalized instrument-verb matrix (values between 0 and 1)
        self.register_buffer('instrument_verb_matrix', torch.tensor([
            # grasper, bipolar, hook,   scissors, clipper, irrigator
            [0.6,     0.15,    0.0,     0.0,     0.0,     0.0],     # grasp
            [1.0,     0.15,    0.15,    0.0,     0.0,     0.15],    # retract
            [0.25,    0.15,    1.0,     0.15,    0.0,     0.15],    # dissect
            [0.0,     1.0,     0.15,    0.05,    0.0,     0.0],     # coagulate
            [0.0,     0.0,     0.0,     0.0,     1.0,     0.0],     # clip
            [0.0,     0.0,     0.1,     1.0,     0.0,     0.0],     # cut
            [0.0,     0.0,     0.0,     0.0,     0.0,     1.0],     # aspirate
            [0.0,     0.0,     0.0,     0.0,     0.0,     1.0],     # irrigate
            [1.0,     0.0,     0.0,     0.0,     0.0,     0.0],     # pack
            [0.15,    0.15,    0.15,    0.15,    0.15,    0.15],    # null_verb
        ], dtype=torch.float32))
        
        # Initialize backbone
        self.backbone = timm.create_model(
            'tf_efficientnet_b4_ns',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        backbone_dim = 1792
        
        # Modified classifier with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),  # Changed from LayerNorm to BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Balanced class weights
        total_weight = 0
        weights = []
        for w in [3.0, 0.8, 0.6, 2.8, 3.2, 3.5, 3.5, 4.0, 5.0, 2.5]:
            total_weight += w
        weights = [w/total_weight * 10 for w in [3.0, 0.8, 0.6, 2.8, 3.2, 3.5, 3.5, 4.0, 5.0, 2.5]]
        
        self.register_buffer('class_weights', torch.FloatTensor(weights))
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        self.validation_step_outputs = []
        self.warmup_epochs = warmup_epochs
    
    def forward(self, x: torch.Tensor, instrument_names: List[str]) -> torch.Tensor:
        """
        Forward pass with soft masking and numerical stability improvements
        """
        instrument_indices = torch.tensor(
            [self.instrument_classes[name] for name in instrument_names],
            device=self.device
        )
        
        visual_features = self.backbone(x)
        logits = self.classifier(visual_features)
        
        # Apply soft masking
        instrument_weights = self.instrument_verb_matrix[:, instrument_indices].t()
        
        # Soft masking with small epsilon
        epsilon = 1e-7
        masked_logits = torch.where(
            instrument_weights > 0,
            logits * instrument_weights,
            logits * epsilon
        )
        
        # Clip extreme values for stability
        masked_logits = torch.clamp(masked_logits, min=-1e7, max=1e7)
        
        return masked_logits
    
    def training_step(self, batch: Tuple[torch.Tensor, List[str], torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with improved stability and logging"""
        images, instrument_names, labels = batch
        logits = self(images, instrument_names)
        
        # Check for numerical instability
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            self.log('warning/unstable_logits', 1.0)
            logits = torch.clamp(logits, min=-1e7, max=1e7)
        
        loss = self.criterion(logits, labels)
        
        # Handle unstable loss
        if torch.isnan(loss) or torch.isinf(loss):
            self.log('warning/unstable_loss', 1.0)
            return torch.tensor(10.0, requires_grad=True, device=self.device)
        
        # Calculate and log metrics
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            acc = (preds == labels).float().mean()
            
            # Log per-class accuracy
            for verb_name, verb_idx in self.verb_classes.items():
                mask = labels == verb_idx
                if mask.sum() > 0:
                    class_acc = (preds[mask] == labels[mask]).float().mean()
                    self.log(f'train_acc_{verb_name}', class_acc, prog_bar=False)
        
        # Log main metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, List[str], torch.Tensor], batch_idx: int) -> None:
        """Validation step with improved metrics"""
        images, instrument_names, labels = batch
        logits = self(images, instrument_names)
        
        # Ensure numerical stability
        logits = torch.clamp(logits, min=-1e7, max=1e7)
        loss = self.criterion(logits, labels)
        
        probs = torch.softmax(logits, dim=1)
        
        self.validation_step_outputs.append({
            'val_loss': loss,
            'probs': probs.detach(),
            'labels': labels.detach()
        })
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            self.log('val_acc', acc, prog_bar=True)
            self.log('val_loss', loss, prog_bar=True, sync_dist=True)
    
    def on_validation_epoch_end(self) -> None:
        """Detailed validation metrics calculation"""
        all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        
        probs_np = all_probs.cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        
        # Calculate metrics for each class
        ap_scores = {}
        for verb_name, verb_idx in self.verb_classes.items():
            binary_labels = (labels_np == verb_idx).astype(np.int32)
            verb_probs = probs_np[:, verb_idx]
            
            ap = average_precision_score(binary_labels, verb_probs)
            ap_scores[verb_name] = ap
            self.log(f'val/AP_{verb_name}', ap, prog_bar=False)
        
        # Log mean metrics
        mean_ap = np.mean(list(ap_scores.values()))
        self.log('val/mAP', mean_ap, prog_bar=True)
        
        preds = np.argmax(probs_np, axis=1)
        metrics = {
            'val/precision': precision_score(labels_np, preds, average='weighted'),
            'val/recall': recall_score(labels_np, preds, average='weighted'),
            'val/f1': f1_score(labels_np, preds, average='weighted')
        }
        
        for name, value in metrics.items():
            self.log(name, value, prog_bar=False)
        
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Optimizer configuration with learning rate scheduling and gradient clipping"""
        optimizer = torch.optim.AdamW([
            {'params': self.backbone.parameters(), 'lr': self.hparams.backbone_learning_rate},
            {'params': self.classifier.parameters(), 'lr': self.hparams.learning_rate}
        ])
        
        # Add gradient clipping directly to the optimizer
        for param_group in optimizer.param_groups:
            param_group['clip_grad_norm'] = 1.0
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_epochs
        )
        
        # Main scheduler
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=50 - self.warmup_epochs,
            eta_min=1e-8
        )
        
        # Chain schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.warmup_epochs]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

    # Remove the on_before_optimizer_step method completely and add this instead:
    def on_train_start(self):
        """Set up gradient clipping at the start of training"""
        self.trainer.gradient_clip_val = 1.0
        self.trainer.gradient_clip_algorithm = "norm"