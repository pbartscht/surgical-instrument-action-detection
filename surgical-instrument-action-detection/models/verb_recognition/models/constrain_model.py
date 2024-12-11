import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import average_precision_score, precision_score, recall_score
import numpy as np
import timm
from typing import Dict, List, Tuple
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

class VerbRecognitionModel(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 9,
        learning_rate: float = 5e-6,
        backbone_learning_rate: float = 5e-7,
        dropout: float = 0.7
    ):
        super().__init__()
        self.save_hyperparameters()

        # Verb Klassen ohne 'pack'
        self.verb_classes = {
        'dissect': 0,    # wie im Dataloader
        'retract': 1,
        'null_verb': 2,
        'coagulate': 3,
        'grasp': 4,
        'clip': 5,
        'aspirate': 6,
        'cut': 7,
        'irrigate': 8
        }
        
        # Einfacher Cross Entropy Loss ohne Gewichtung
        self.criterion = nn.CrossEntropyLoss()
        
        # Backbone
        self.backbone = timm.create_model(
            'tf_efficientnet_b0_ns',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        backbone_dim = 1280
        
        # Classifier
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
        
        # Validation Metrics Storage
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Vereinfachter Forward Pass ohne Constraint Matrix
        """
        visual_features = self.backbone(x)
        logits = self.classifier(visual_features)
        probs = torch.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'probabilities': probs
        }

    def training_step(self, batch: Tuple[torch.Tensor, List[str], torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training Step ohne zusätzliche Präprozessierung"""
        images, _, labels = batch  # Instrument names werden nicht mehr benötigt
        outputs = self(images)
        loss = self.criterion(outputs['logits'], labels)
        
        # Logging
        self.log('train/loss', loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, List[str], torch.Tensor], batch_idx: int) -> None:
        """Validation Step"""
        images, _, labels = batch
        outputs = self(images)
        
        val_loss = self.criterion(outputs['logits'], labels)
        
        # Speichere relevante Outputs für epoch_end
        self.validation_step_outputs.append({
            'val_loss': val_loss,
            'probs': outputs['probabilities'].detach(),
            'labels': labels.detach()
        })
        
        # Basis-Metriken direkt loggen
        self.log('val/loss', val_loss, prog_bar=True)
    
    def on_validation_epoch_end(self) -> None:
        """Berechnung von AP, Precision und Recall Metriken"""
        # Sammle alle Outputs
        all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        
        # Predictions für Precision/Recall
        all_preds = torch.argmax(all_probs, dim=1)
        
        # Berechne Metriken für jede Klasse
        aps = []
        precisions = []
        recalls = []
        
        for verb_name, verb_idx in self.verb_classes.items():
            # Binäre Labels für diese Klasse
            binary_labels = (all_labels == verb_idx).cpu().numpy()
            verb_probs = all_probs[:, verb_idx].cpu().numpy()
            binary_preds = (all_preds == verb_idx).cpu().numpy()
            
            # Berechne Metriken nur wenn die Klasse im Validation Set vorkommt
            if binary_labels.sum() > 0:
                ap = average_precision_score(binary_labels, verb_probs)
                prec = precision_score(binary_labels, binary_preds, zero_division=0)
                rec = recall_score(binary_labels, binary_preds, zero_division=0)
                
                # Logging
                self.log(f'val/AP_{verb_name}', ap, prog_bar=False)
                #self.log(f'val/precision_{verb_name}', prec, prog_bar=False)
                #self.log(f'val/recall_{verb_name}', rec, prog_bar=False)
                
                aps.append(ap)
                precisions.append(prec)
                recalls.append(rec)
                
                # F1-Score
                f1 = 2 * (prec * rec) / (prec + rec + 1e-10)
                #self.log(f'val/f1_{verb_name}', f1, prog_bar=False)
        
        # Durchschnitte
        self.log('val/mAP', np.mean(aps), prog_bar=True)
        self.log('val/mean_precision', np.mean(precisions), prog_bar=True)
        self.log('val/mean_recall', np.mean(recalls), prog_bar=True)
        
        # Global weighted Metriken
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
        
        self.log('val/weighted_precision', weighted_precision, prog_bar=True)
        self.log('val/weighted_recall', weighted_recall, prog_bar=True)
        
        # Clear saved outputs
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.backbone.parameters(), 'lr': self.hparams.backbone_learning_rate},
            {'params': self.classifier.parameters(), 'lr': self.hparams.learning_rate}
        ])
        
        # Linear Warmup + Cosine Decay
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-7)
        
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