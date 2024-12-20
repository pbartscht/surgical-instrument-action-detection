import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
import numpy as np
import timm
from typing import Dict, List, Tuple
from torch.optim.lr_scheduler import CosineAnnealingLR

class SimpleVerbRecognition(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 10,
        learning_rate: float = 5e-5,
        backbone_learning_rate: float = 5e-6,
        dropout: float = 0.5
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
        
        self.register_buffer('instrument_verb_matrix', torch.tensor([
            # grasper, bipolar, hook,  scissors, clipper, irrigator
            [1.2, 0.8, 0.2, 0.0, 0.0, 0.0],    # grasp    (erhöht für grasper)
            [1.2, 0.6, 0.2, 0.0, 0.0, 0.8],    # retract  (angepasst für häufige Kombinationen)
            [0.6, 0.8, 1.2, 0.8, 0.0, 0.8],    # dissect  (balanciert)
            [0.0, 1.2, 0.6, 0.8, 0.0, 0.0],    # coagulate (verstärkt für bipolar)
            [0.0, 0.0, 0.0, 0.0, 1.2, 0.0],    # clip     (verstärkt für clipper)
            [0.0, 0.0, 0.0, 1.2, 0.0, 0.0],    # cut      (verstärkt für scissors)
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.2],    # aspirate (verstärkt für irrigator)
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.2],    # irrigate (verstärkt für irrigator)
            [1.2, 0.0, 0.0, 0.0, 0.0, 0.0],    # pack     (verstärkt für grasper)
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],    # null_verb (reduzierte Gewichte)
        ], dtype=torch.float32))
        
        self.backbone = timm.create_model(
            'tf_efficientnet_b4_ns',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        backbone_dim = 1792
        
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Angepasste Klassengewichte basierend auf inversen Frequenzen
        self.register_buffer('class_weights', torch.FloatTensor([
            # Gewichte basierend auf 1/frequency
            3.0,    # grasp     (3.28%)  -> höheres Gewicht
            0.8,    # retract   (32.82%) -> niedrigeres Gewicht
            0.6,    # dissect   (47.99%) -> niedrigstes Gewicht
            2.8,    # coagulate (4.53%)  -> höheres Gewicht
            3.2,    # clip      (2.92%)  -> höheres Gewicht
            3.5,    # cut       (1.45%)  -> höheres Gewicht
            3.5,    # aspirate  (1.66%)  -> höheres Gewicht
            4.0,    # irrigate  (0.30%)  -> sehr hohes Gewicht
            5.0,    # pack      (0.01%)  -> höchstes Gewicht
            2.5     # null_verb (5.04%)  -> mittleres Gewicht
        ]))
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        self.validation_step_outputs = []
    
    def forward(self, x: torch.Tensor, instrument_names: List[str]) -> torch.Tensor:
        instrument_indices = torch.tensor(
            [self.instrument_classes[name] for name in instrument_names],
            device=self.device
        )
        
        visual_features = self.backbone(x)
        logits = self.classifier(visual_features)
        
        instrument_weights = self.instrument_verb_matrix[:, instrument_indices].t()
        instrument_weights = instrument_weights / instrument_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        
        weighted_logits = logits * instrument_weights
        
        return weighted_logits
    
    def training_step(self, batch: Tuple[torch.Tensor, List[str], torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with improved logging"""
        images, instrument_names, labels = batch
        logits = self(images, instrument_names)
        loss = self.criterion(logits, labels)
        
        # Calculate and log accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            
            # Calculate per-class accuracy
            for verb_name, verb_idx in self.verb_classes.items():
                mask = labels == verb_idx
                if mask.sum() > 0:
                    class_acc = (preds[mask] == labels[mask]).float().mean()
                    self.log(f'train_acc_{verb_name}', class_acc, prog_bar=False)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, List[str], torch.Tensor], batch_idx: int) -> None:
        """Validation step with detailed metrics"""
        images, instrument_names, labels = batch
        logits = self(images, instrument_names)
        loss = self.criterion(logits, labels)
        
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Store outputs for epoch end processing
        self.validation_step_outputs.append({
            'val_loss': loss,
            'probs': probs.detach(),
            'labels': labels.detach()
        })
        
        # Immediate metric logging
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            self.log('val_acc', acc, prog_bar=True)
            self.log('val_loss', loss, prog_bar=True, sync_dist=True)
    
    def on_validation_epoch_end(self) -> None:
        """Calculate and log detailed AP metrics for each verb at epoch end"""
        all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        
        probs_np = all_probs.cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        
        # Calculate AP for each verb class
        ap_scores = {}
        for verb_name, verb_idx in self.verb_classes.items():
            binary_labels = (labels_np == verb_idx).astype(np.int32)
            verb_probs = probs_np[:, verb_idx]
            
            ap = average_precision_score(binary_labels, verb_probs)
            ap_scores[verb_name] = ap
            
            self.log(f'val/AP_{verb_name}', ap, prog_bar=False)
        
        # Calculate and store mean AP
        mean_ap = np.mean(list(ap_scores.values()))
        self.log('val/mAP', mean_ap, prog_bar=True)
        
        # Calculate overall metrics
        preds = np.argmax(probs_np, axis=1)
        metrics = {
            'val/precision': precision_score(labels_np, preds, average='weighted'),
            'val/recall': recall_score(labels_np, preds, average='weighted'),
            'val/f1': f1_score(labels_np, preds, average='weighted')
        }
        
        for name, value in metrics.items():
            self.log(name, value, prog_bar=False)
        
        # Clear saved outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Optimizer configuration with learning rate scheduling"""
        optimizer = torch.optim.AdamW([
            {'params': self.backbone.parameters(), 'lr': self.hparams.backbone_learning_rate},
            {'params': self.classifier.parameters(), 'lr': self.hparams.learning_rate}
        ])
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-8
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }