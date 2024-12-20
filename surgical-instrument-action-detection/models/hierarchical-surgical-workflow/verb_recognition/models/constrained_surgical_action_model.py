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
        
        # Definiere valide Instrument-Verb Paare
        self.VALID_PAIRS = {
            'grasper': ['grasp', 'retract', 'null_verb'],
            'hook': ['dissect', 'cut', 'null_verb', 'coagulate'],
            'bipolar': ['coagulate', 'dissect', 'null_verb'],
            'clipper': ['clip', 'null_verb'],
            'scissors': ['cut', 'null_verb'],
            'irrigator': ['aspirate', 'irrigate', 'null_verb']
        }

        # Verb Klassen
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
        
        # Instrument Klassen
        self.instrument_classes = {
            'grasper': 0,
            'bipolar': 1,
            'hook': 2,
            'scissors': 3,
            'clipper': 4,
            'irrigator': 5
        }
        
        # Erstelle Constraint Matrix für Instrument-Verb Kombinationen
        self.register_buffer('constraint_matrix', self._create_constraint_matrix())
        
        # Basis Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Backbone
        self.backbone = timm.create_model(
            'tf_efficientnet_b0_ns',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        backbone_dim = 1280
        
        # Instrument Embedding
        self.instrument_embedding = nn.Embedding(
            num_embeddings=len(self.instrument_classes),
            embedding_dim=256
        )
        
        # Classifier mit Instrument-Kontext
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim + 256, 512),  # +256 für Instrument Embedding
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

    def _create_constraint_matrix(self) -> torch.Tensor:
        """Erstellt eine binäre Matrix für valide Instrument-Verb Kombinationen"""
        matrix = torch.zeros((len(self.instrument_classes), len(self.verb_classes)))
        
        for instrument, verbs in self.VALID_PAIRS.items():
            instrument_idx = self.instrument_classes[instrument]
            for verb in verbs:
                verb_idx = self.verb_classes[verb]
                matrix[instrument_idx, verb_idx] = 1.0
                
        return matrix

    def forward(self, x: torch.Tensor, instrument_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward Pass mit Instrument-Kontext und Constraints
        
        Args:
            x: Bild Tensor
            instrument_indices: Tensor mit Instrument-Indizes
        """
        # Visual Features
        visual_features = self.backbone(x)
        
        # Instrument Embedding
        instrument_features = self.instrument_embedding(instrument_indices)
        
        # Kombiniere Features
        combined_features = torch.cat([visual_features, instrument_features], dim=1)
        
        # Classifier
        logits = self.classifier(combined_features)
        
        # Anwenden der Constraints
        batch_constraints = self.constraint_matrix[instrument_indices]  # [B, num_verbs]
        masked_logits = logits * batch_constraints
        masked_logits = masked_logits + (1 - batch_constraints) * -1e9  # Setze ungültige zu -inf
        
        # Probabilities
        probs = torch.softmax(masked_logits, dim=-1)
        
        return {
            'logits': masked_logits,
            'probabilities': probs
        }

    def training_step(self, batch: Tuple[torch.Tensor, List[str], torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training Step mit Instrument-Kontext"""
        images, instrument_names, labels = batch
        
        # Konvertiere Instrument-Namen zu Indizes
        instrument_indices = torch.tensor([
            self.instrument_classes[name] for name in instrument_names
        ], device=self.device)
        
        outputs = self(images, instrument_indices)
        loss = self.criterion(outputs['logits'], labels)
        
        # Logging
        self.log('train/loss', loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, List[str], torch.Tensor], batch_idx: int) -> None:
        """Validation Step mit Instrument-Kontext"""
        images, instrument_names, labels = batch
        
        # Konvertiere Instrument-Namen zu Indizes
        instrument_indices = torch.tensor([
            self.instrument_classes[name] for name in instrument_names
        ], device=self.device)
        
        outputs = self(images, instrument_indices)
        val_loss = self.criterion(outputs['logits'], labels)
        
        # Speichere Outputs für epoch_end
        self.validation_step_outputs.append({
            'val_loss': val_loss,
            'probs': outputs['probabilities'].detach(),
            'labels': labels.detach(),
            'instrument_indices': instrument_indices.detach()
        })
        
        self.log('val/loss', val_loss, prog_bar=True)
    
    def on_validation_epoch_end(self) -> None:
        """Berechnet Metriken pro Instrument-Verb Kombination"""
        all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        all_instruments = torch.cat([x['instrument_indices'] for x in self.validation_step_outputs])
        
        all_preds = torch.argmax(all_probs, dim=1)
        
        # Metriken pro Instrument-Verb Paar
        aps = []
        precisions = []
        recalls = []
        
        for instrument_name, instrument_idx in self.instrument_classes.items():
            valid_verbs = self.VALID_PAIRS[instrument_name]
            instrument_mask = (all_instruments == instrument_idx)
            
            if instrument_mask.sum() > 0:
                for verb in valid_verbs:
                    verb_idx = self.verb_classes[verb]
                    
                    # Erstelle binäre Labels für dieses Paar
                    pair_mask = instrument_mask & (all_labels == verb_idx)
                    if pair_mask.sum() > 0:
                        binary_labels = pair_mask.cpu().numpy()
                        pair_probs = all_probs[instrument_mask, verb_idx].cpu().numpy()
                        binary_preds = (all_preds[instrument_mask] == verb_idx).cpu().numpy()
                        
                        ap = average_precision_score(binary_labels, pair_probs)
                        prec = precision_score(binary_labels, binary_preds, zero_division=0)
                        rec = recall_score(binary_labels, binary_preds, zero_division=0)
                        
                        pair_name = f"{instrument_name}-{verb}"
                        self.log(f'val/AP_{pair_name}', ap, prog_bar=False)
                        
                        aps.append(ap)
                        precisions.append(prec)
                        recalls.append(rec)
        
        # Durchschnittliche Metriken
        self.log('val/mAP', np.mean(aps), prog_bar=True)
        self.log('val/mean_precision', np.mean(precisions), prog_bar=True)
        self.log('val/mean_recall', np.mean(recalls), prog_bar=True)
        
        # Globale gewichtete Metriken
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
        
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # Unterschiedliche Lernraten für verschiedene Komponenten
        optimizer = torch.optim.AdamW([
            {'params': self.backbone.parameters(), 'lr': self.hparams.backbone_learning_rate},
            {'params': self.instrument_embedding.parameters(), 'lr': self.hparams.learning_rate},
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