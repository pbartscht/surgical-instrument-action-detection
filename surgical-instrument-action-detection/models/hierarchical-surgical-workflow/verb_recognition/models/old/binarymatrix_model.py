import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
import numpy as np
import timm
from typing import Dict, List, Tuple
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

class VerbRecognitionModel(pl.LightningModule):
    """
    PyTorch Lightning Modell für Verb-Klassifikation in chirurgischen Videos.
    Nutzt sowohl visuelle Features aus Instrument-Crops als auch explizite Instrumenten-Information.
    
    Attributes:
        num_classes (int): Anzahl der Verb-Klassen (default: 10)
        learning_rate (float): Learning Rate für den Classifier
        backbone_learning_rate (float): Learning Rate für das Backbone
        dropout (float): Dropout-Rate
        warmup_epochs (int): Anzahl der Warmup-Epochen
    """
    def __init__(
        self,
        num_classes: int = 10,
        learning_rate: float = 1e-5,
        backbone_learning_rate: float = 1e-6,
        dropout: float = 0.5,
        warmup_epochs: int = 5
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Definition der Klassen-Mappings (entsprechend DataLoader)
        self.verb_classes = {
            'grasp': 0, 'retract': 1, 'dissect': 2, 'coagulate': 3,
            'clip': 4, 'cut': 5, 'aspirate': 6, 'irrigate': 7,
            'pack': 8, 'null_verb': 9
        }
        
        self.instrument_classes = {
            'grasper': 0, 'bipolar': 1, 'hook': 2,
            'scissors': 3, 'clipper': 4, 'irrigator': 5
        }
        
        # Binäre Instrument-Verb Constraints Matrix basierend auf Datenanalyse
        self.register_buffer('instrument_verb_matrix', torch.tensor([
            # grasper bipolar hook scissors clipper irrigator
            [1,      0,      0,   0,       0,      0],  # grasp
            [1,      1,      1,   0,       0,      0],  # retract 
            [1,      1,      1,   1,       0,      0],  # dissect
            [0,      1,      1,   0,       0,      0],  # coagulate
            [0,      0,      0,   0,      1,       0],  # clip
            [0,      0,      0,   1,       0,      0],  # cut
            [0,      0,      0,   0,       0,      1],  # aspirate
            [0,      0,      0,   0,       0,      1],  # irrigate
            [1,      0,      0,   0,       0,      0],  # pack
            [1,      1,      1,   1,       1,      1],  # null_verb
        ], dtype=torch.float32))
        
        # EfficientNet Backbone für visuelle Feature-Extraktion
        self.backbone = timm.create_model(
            'tf_efficientnet_b4_ns',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        backbone_dim = 1792
        
        # One-Hot Encoding Dimension für Instrumente
        instrument_dim = len(self.instrument_classes)
        
        # Classifier mit kombinierter Feature-Verarbeitung
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim + instrument_dim, 512),
            nn.LayerNorm(512),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Klassengewichte für unbalancierte Daten
        weights = [3.0, 0.8, 0.6, 2.8, 3.2, 3.5, 3.5, 4.0, 5.0, 2.5]
        total_weight = sum(weights)
        weights = [w / total_weight * 10 for w in weights]
        self.register_buffer('class_weights', torch.FloatTensor(weights))
        
        # Loss Function
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Validation Metrics Storage
        self.validation_step_outputs = []
        self.warmup_epochs = warmup_epochs
        
        # Confidence Threshold für Metriken
        self.confidence_threshold = 0.7
    
    def forward(self, x: torch.Tensor, instrument_names: List[str]) -> torch.Tensor:
        """
        Forward Pass mit verbesserter numerischer Stabilität
        """
        # One-Hot Encoding der Instrumente
        batch_size = x.shape[0]
        instrument_one_hot = torch.zeros(
            batch_size, 
            len(self.instrument_classes), 
            device=self.device
        )
        for i, name in enumerate(instrument_names):
            instrument_one_hot[i, self.instrument_classes[name]] = 1
        
        # Visual Features
        visual_features = self.backbone(x)
        
        # Kombiniere Features
        combined_features = torch.cat([visual_features, instrument_one_hot], dim=1)
        
        # Basis-Logits
        logits = self.classifier(combined_features)
        
        # Instrument-Constraints mit weicherem Masking
        instrument_indices = torch.tensor(
            [self.instrument_classes[name] for name in instrument_names],
            device=self.device
        )
        instrument_constraints = self.instrument_verb_matrix[:, instrument_indices].t()
        
        # Softeres Masking mit kontrolliertem Minimum
        masked_logits = torch.where(
            instrument_constraints > 0,
            logits,
            torch.full_like(logits, -20.0)  # Definierter Minimalwert statt -inf
        )
        
        return masked_logits
    
    def training_step(self, batch: Tuple[torch.Tensor, List[str], torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Stabilerer Training Step
        """
        images, instrument_names, labels = batch
        logits = self(images, instrument_names)
        
        # Numerische Stabilität für Softmax
        logits = logits - logits.max(dim=1, keepdim=True)[0]  # LogSumExp Trick
        
        # Primary Loss mit Stabilität
        primary_loss = self.criterion(logits, labels)
        
        # Berechne Softmax mit numerischer Stabilität
        probs = torch.softmax(logits, dim=1)
        
        # Instrument Constraints
        instrument_indices = torch.tensor(
            [self.instrument_classes[name] for name in instrument_names],
            device=self.device
        )
        allowed_verbs = self.instrument_verb_matrix[:, instrument_indices].t()
        
        # Sanftere Auxiliary Loss
        invalid_probs = probs * (1 - allowed_verbs)
        aux_loss = torch.mean(torch.sum(invalid_probs, dim=1)) * 0.1  # Reduzierter Faktor
        
        # Gesamtverlust mit Clipping
        total_loss = primary_loss + aux_loss
        
        # Logging
        self.log('train/primary_loss', primary_loss.item(), prog_bar=True)
        self.log('train/aux_loss', aux_loss.item(), prog_bar=True)
        
        return total_loss
   
    def validation_step(self, batch: Tuple[torch.Tensor, List[str], torch.Tensor], batch_idx: int) -> None:
       """
       Validation Step mit detaillierten Metriken
       """
       images, instrument_names, labels = batch
       logits = self(images, instrument_names)
       
       # Berechne Loss
       loss = self.criterion(logits, labels)
       
       # Berechne Probabilities
       probs = torch.softmax(logits, dim=1)
       
       # Sammle Outputs für Epoch End
       self.validation_step_outputs.append({
           'val_loss': loss,
           'probs': probs.detach(),
           'labels': labels.detach()
       })
       
       # Logging während des Steps
       with torch.no_grad():
           preds = torch.argmax(logits, dim=1)
           acc = (preds == labels).float().mean()
           self.log('val_acc', acc, prog_bar=True)
           self.log('val_loss', loss, prog_bar=True, sync_dist=True)
   
    def on_validation_epoch_end(self) -> None:
        """
        Berechnung der Validation Metriken am Ende der Epoch
        """
        all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        
        probs_np = all_probs.cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        average_precisions = []

        # Berechne Average Precision für jede Klasse
        for verb_name, verb_idx in self.verb_classes.items():
            binary_labels = (labels_np == verb_idx).astype(np.int32)
            verb_probs = probs_np[:, verb_idx]
            ap = average_precision_score(binary_labels, verb_probs)
            self.log(f'val/AP_{verb_name}', ap, prog_bar=False)
            average_precisions.append(ap)
        
        # Berechne mean Average Precision (mAP)
        mAP = np.mean(average_precisions)
    
        # Berechne weitere Metriken
        preds = np.argmax(probs_np, axis=1)
        metrics = {
            'val/precision': precision_score(labels_np, preds, average='weighted'),
            'val/recall': recall_score(labels_np, preds, average='weighted'),
            'val/f1': f1_score(labels_np, preds, average='weighted'),
            'val/mAP': mAP
        }
        
        for name, value in metrics.items():
            self.log(name, value, prog_bar=False)
        
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """
        Optimizer-Konfiguration mit Gradient Clipping
        """
        optimizer = torch.optim.AdamW([
            {'params': self.backbone.parameters(), 'lr': self.hparams.backbone_learning_rate},
            {'params': self.classifier.parameters(), 'lr': self.hparams.learning_rate}
        ])
        
        # Gradient Clipping
        for param_group in optimizer.param_groups:
            param_group['clip_grad_norm'] = 1.0
        
        # Scheduler Setup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,  # Sanfterer Start
            end_factor=1.0,
            total_iters=self.warmup_epochs
        )
        
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=50 - self.warmup_epochs,
            eta_min=1e-8
        )
        
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