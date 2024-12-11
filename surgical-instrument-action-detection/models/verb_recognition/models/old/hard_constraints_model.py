import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
import numpy as np
import timm
from typing import Dict, List, Tuple
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from pytorch_lightning.callbacks import EarlyStopping

class VerbRecognitionModel(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 10,
        learning_rate: float = 5e-6,
        backbone_learning_rate: float = 5e-7,
        dropout: float = 0.7
    ):
        super().__init__()
        self.save_hyperparameters()

        # Verb Klassen hinzufügen
        self.verb_classes = {
            'grasp': 0, 'retract': 1, 'dissect': 2, 'coagulate': 3,
            'clip': 4, 'cut': 5, 'aspirate': 6, 'irrigate': 7,
            'pack': 8, 'null_verb': 9
        }
        
        # Class Weights basierend auf der Verteilung
        class_weights = torch.tensor([
            5.8,    # grasp     (3000 samples)
            0.7,    # retract   (25000 samples)
            0.5,    # dissect   (35000 samples)
            4.4,    # coagulate (4000 samples)
            7.0,    # clip      (2500 samples)
            9.0,    # cut       (1000 samples)
            8.2,    # aspirate  (1500 samples)
            10.0,   # irrigate  (sehr selten)
            10.0,   # pack      (sehr selten)
            3.5,    # null_verb (5000 samples)
        ], dtype=torch.float32)
        
        # Registriere die Gewichte als Buffer
        self.register_buffer('class_weights', class_weights)
        
        # Weighted Loss Function
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        self.register_buffer('instrument_verb_matrix', torch.tensor([
        # grasper, bipolar, hook, scissors, clipper, irrigator
        [1,      0,      0,   0,       0,      0],  # grasp
        [1,      0,      0,   0,       0,      0],  # retract
        [0,      0,      1,   1,       0,      0],  # dissect
        [0,      1,      1,   0,       0,      0],  # coagulate
        [0,      0,      0,   0,      1,       0],  # clip
        [0,      0,      0,   1,       0,      0],  # cut
        [0,      0,      0,   0,       0,      1],  # aspirate
        [0,      0,      0,   0,       0,      1],  # irrigate
        [0,      0,      0,   0,       0,      0],  # pack
        [1,      1,      1,   1,       1,      1],  # null_verb
        ], dtype=torch.float32))

        # Instrument Klassen
        self.instrument_classes = {
            'grasper': 0, 'bipolar': 1, 'hook': 2,
            'scissors': 3, 'clipper': 4, 'irrigator': 5
        }
        
        # Backbone
        self.backbone = timm.create_model(
            'tf_efficientnet_b0_ns',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        backbone_dim = 1280
        
        # Vereinfachter Classifier ohne separate Instrument-Verarbeitung
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
        # Validation Metrics Storage hinzufügen
        self.validation_step_outputs = []

    def get_verb_mask(self, instrument_name: str) -> torch.Tensor:
        """
        Erstellt eine binäre Maske für erlaubte Verben basierend auf dem Instrument
        """
        instrument_idx = self.instrument_classes[instrument_name]
        return self.instrument_verb_matrix[:, instrument_idx]

    def forward(self, x: torch.Tensor, instrument_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward Pass mit direkter Anwendung der Instrument-Constraints
        """
        # 1. Visuelle Feature-Extraktion
        visual_features = self.backbone(x)
        
        # 2. Verb-Logits berechnen
        logits = self.classifier(visual_features)
        
        # 3. Instrument-spezifische Masken erstellen
        batch_masks = []
        for name in instrument_names:
            mask = self.get_verb_mask(name)
            batch_masks.append(mask)
        instrument_masks = torch.stack(batch_masks)
        
        # 4. Logits maskieren (-10000.0 für nicht erlaubte Verben)
        masked_logits = torch.where(
            instrument_masks > 0,
            logits,
            torch.full_like(logits, -10000.0)
        )
        
        # 5. Softmax nur über erlaubte Verben
        probs = self.masked_softmax(masked_logits, instrument_masks)
        
        return {
            'logits': logits,
            'masked_logits': masked_logits,
            'probabilities': probs,
            'valid_verb_mask': instrument_masks
        }
    
    def masked_softmax(self, logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Numerisch stabile Version von masked softmax"""
        masked_logits = logits.clone()
        masked_logits[mask == 0] = -10000.0
        
        # Numerische Stabilität
        max_logits = torch.max(masked_logits, dim=dim, keepdim=True)[0]
        exp_logits = torch.exp(masked_logits - max_logits)
        exp_logits = exp_logits * mask
        
        sum_exp = torch.sum(exp_logits, dim=dim, keepdim=True)
        sum_exp = torch.clamp(sum_exp, min=1e-10)
        
        return exp_logits / sum_exp

    def preprocess_batch(self, batch: Tuple[torch.Tensor, List[str], torch.Tensor]) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Filtert invalide Instrument-Verb Kombinationen aus dem Batch
        """
        images, instrument_names, labels = batch
        valid_samples = []
        valid_instruments = []
        valid_labels = []
        
        for i, (label, name) in enumerate(zip(labels, instrument_names)):
            mask = self.get_verb_mask(name)
            if mask[label]:  # Wenn die Kombination valide ist
                valid_samples.append(images[i])
                valid_instruments.append(name)
                valid_labels.append(label)
        
        # Wenn keine validen Samples gefunden wurden
        if not valid_samples:
            return None, None, None
            
        # Konvertiere zurück zu Tensor/List
        valid_images = torch.stack(valid_samples)
        valid_labels = torch.tensor(valid_labels, device=labels.device, dtype=labels.dtype)
        
        return valid_images, valid_instruments, valid_labels

    def training_step(self, batch: Tuple[torch.Tensor, List[str], torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training Step mit Präprozessierung"""
        # Präprozessiere den Batch
        images, instrument_names, labels = self.preprocess_batch(batch)
        
        # Wenn der gesamte Batch invalid war
        if images is None:
            # Returne einen Zero-Loss mit grad, damit Lightning nicht crashed
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # Normal forward pass mit validen Samples
        outputs = self(images, instrument_names)
        masked_logits = outputs['masked_logits']
        loss = self.criterion(masked_logits, labels)
        
        # Optional: Tracke die Anzahl der verworfenen Samples
        original_batch_size = len(batch[0])
        valid_batch_size = len(images)
        invalid_ratio = 1 - (valid_batch_size / original_batch_size)
        
        # Logging
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/invalid_sample_ratio', invalid_ratio, prog_bar=True)
        
        return loss
    def validation_step(self, batch: Tuple[torch.Tensor, List[str], torch.Tensor], batch_idx: int) -> None:
        """Validation Step mit Speicherung für epoch_end Metriken"""
        images, instrument_names, labels = batch
        outputs = self(images, instrument_names)
        
        masked_logits = outputs['masked_logits']
        val_loss = self.criterion(masked_logits, labels)
        
        # Speichere relevante Outputs für epoch_end
        self.validation_step_outputs.append({
            'val_loss': val_loss,
            'probs': outputs['probabilities'].detach(),
            'labels': labels.detach(),
            'instrument_names': instrument_names  # Für mögliche Instrument-spezifische Analysen
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
                # Average Precision
                ap = average_precision_score(binary_labels, verb_probs)
                
                # Precision und Recall
                prec = precision_score(binary_labels, binary_preds, zero_division=0)
                rec = recall_score(binary_labels, binary_preds, zero_division=0)
                
                # Logging
                self.log(f'val/AP_{verb_name}', ap, prog_bar=False)
                self.log(f'val/precision_{verb_name}', prec, prog_bar=False)
                self.log(f'val/recall_{verb_name}', rec, prog_bar=False)
                
                # Sammeln für Durchschnitte
                aps.append(ap)
                precisions.append(prec)
                recalls.append(rec)
                
                # Optional: F1-Score pro Klasse
                f1 = 2 * (prec * rec) / (prec + rec + 1e-10)
                self.log(f'val/f1_{verb_name}', f1, prog_bar=False)
        
        # Berechne und logge Durchschnitte
        mAP = np.mean(aps)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        
        # Logging der Hauptmetriken
        self.log('val/mAP', mAP, prog_bar=True)
        self.log('val/mean_precision', mean_precision, prog_bar=True)
        self.log('val/mean_recall', mean_recall, prog_bar=True)
        
        # Global (weighted) Metriken
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
        main_scheduler = CosineAnnealingLR(optimizer, T_max=45, eta_min=1e-7)
        
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