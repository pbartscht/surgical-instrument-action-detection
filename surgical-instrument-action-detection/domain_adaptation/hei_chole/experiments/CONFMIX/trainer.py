import torch
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from ultralytics import YOLO
from base_setup import TOOL_MAPPING, IMAGE_SIZE
from confmix_core import ConfidenceBasedDetector, ConfMixDetector
from dataset_loader import create_confmix_dataloader

class ConfMixTrainer:
    def __init__(self, inference_weights):
        """Initialisiert den ConfMix Trainer"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model Setup
        self.model = YOLO(inference_weights)
        self.model = self.model.to(self.device)

        self.confidence_detector = ConfidenceBasedDetector(self.model)
        self.confmix_detector = ConfMixDetector(self.confidence_detector)

        # Optimizer Setup ähnlich wie in YOLOTrainer
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
        
        # Parameter Gruppierung wie in YOLOTrainer
        for module_name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:
                    g[2].append(param)
                elif isinstance(module, bn):
                    g[1].append(param)
                else:
                    g[0].append(param)
        
        # Optimizer mit Parameter Gruppen
        self.optimizer = torch.optim.Adam(g[2], lr=0.001)  # bias params
        self.optimizer.add_param_group({'params': g[0], 'weight_decay': 0.0005})  # weight params with decay
        self.optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # weight params without decay
        
        # Training Settings
        self.num_epochs = 50
        self.batch_size = 8
        self.save_frequency = 5
        
        # ConfMix spezifische Parameter
        self.pseudo_label_threshold = 0.25  # Start-Threshold
        self.consistency_weight = 0.5
        self.class_weights = self._initialize_class_weights()
        
        # Performance Tracking
        self.class_performance = {
            class_id: {'tp': 0, 'fp': 0, 'fn': 0} 
            for class_id in TOOL_MAPPING.keys()
        }
        
        # Loss History
        self.loss_history = []
        
    def train(self, source_path, target_path, save_dir):
        """Haupttraining Loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting training with source: {source_path}, target: {target_path}")
        
        # DataLoader Setup
        dataloader = create_confmix_dataloader(
            source_path=source_path,
            target_path=target_path,
            confmix_detector=self.confmix_detector,
            batch_size=self.batch_size
        )
        
        # Training Loop
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Update pseudo-label threshold progressively
            self.pseudo_label_threshold = self._get_progressive_threshold(epoch)
            
            # Train one epoch
            epoch_loss = self._train_epoch(dataloader)
            
            # Update class weights based on performance
            self._update_class_weights()
            
            print(f"Epoch {epoch+1} - Average Loss: {epoch_loss:.4f}")
            self._print_class_performance()
            
            # Save checkpoints
            if (epoch + 1) % self.save_frequency == 0:
                self._save_checkpoint(save_dir / f"epoch_{epoch+1}")
                
            self.loss_history.append(epoch_loss)
    
    def _train_epoch(self, dataloader):
        """Trainiert eine einzelne Epoche"""
        epoch_losses = []
        self._reset_class_performance()
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass durch YOLO
                predictions = self.model(batch['mixed_images'])
                
                # Berechne Loss
                loss = self.compute_confmix_loss(
                    predictions=predictions,
                    source_labels=batch['source_labels'],
                    mixed_labels=batch['mixed_labels'],
                    masks=batch['mixing_mask']
                )
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                epoch_losses.append(loss.item())
                
                # Update Performance Tracking
                if (batch_idx + 1) % 10 == 0:
                    self._update_class_performance(predictions, batch)
                    print(f"Batch {batch_idx+1}/{len(dataloader)} - "
                          f"Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        return np.mean(epoch_losses) if epoch_losses else 0
    
    def compute_confmix_loss(self, predictions, source_labels, mixed_labels, masks):
        """Berechnet den ConfMix-spezifischen Loss"""
        batch_size = len(predictions)
        total_loss = 0
        
        for idx in range(batch_size):
            pred = predictions[idx]
            mask = masks[idx]
            
            # Source Region Loss (mask == 0)
            source_region = ~mask.bool()
            if source_region.any():
                source_loss = self.model.criterion(
                    pred[source_region],
                    self._format_yolo_labels(source_labels[idx])
                )
                total_loss += source_loss
            
            # Target Region Loss (mask == 1)
            target_region = mask.bool()
            if target_region.any():
                confident_labels = self._filter_confident_labels(
                    mixed_labels[idx], 
                    self.pseudo_label_threshold
                )
                if confident_labels:
                    # Gewichte Loss basierend auf Klassengewichten
                    pseudo_loss = 0
                    for label in confident_labels:
                        class_weight = self.class_weights[label['class']]
                        class_loss = self.model.criterion(
                            pred[target_region],
                            self._format_yolo_labels([label])
                        )
                        pseudo_loss += class_weight * class_loss
                    
                    total_loss += self.consistency_weight * pseudo_loss
        
        return total_loss / batch_size
    
    def _format_yolo_labels(self, labels):
        """Konvertiert Labels in YOLO-kompatibles Format"""
        if not labels:
            return torch.zeros((0, 6), device=self.device)
        
        formatted = []
        for label in labels:
            formatted.append([
                0,  # batch_idx
                label['class'],
                *label['box']  # x, y, w, h
            ])
        return torch.tensor(formatted, device=self.device)
    
    def _filter_confident_labels(self, labels, threshold):
        """Filtert Labels basierend auf Confidence"""
        return [label for label in labels 
                if label.get('confidence', 0) > threshold]
    
    def _get_progressive_threshold(self, epoch):
        """Berechnet progressiven Threshold basierend auf Trainingfortschritt"""
        progress = epoch / self.num_epochs
        base_threshold = 0.25
        max_threshold = 0.5
        return base_threshold + (max_threshold - base_threshold) * progress
    
    def _initialize_class_weights(self):
        """Initialisiert Klassengewichte basierend auf TOOL_MAPPING"""
        return {class_id: info['weight'] 
                for class_id, info in TOOL_MAPPING.items()}
    
    def _update_class_weights(self):
        """Aktualisiert Klassengewichte basierend auf Performance"""
        for class_id, perf in self.class_performance.items():
            if perf['tp'] + perf['fp'] + perf['fn'] > 0:
                f1 = 2 * perf['tp'] / (2 * perf['tp'] + perf['fp'] + perf['fn'])
                # Erhöhe Gewicht für schlecht performende Klassen
                self.class_weights[class_id] = max(1.0, 1.5 - f1)
    
    def _reset_class_performance(self):
        """Setzt Performance-Tracking zurück"""
        for class_id in self.class_performance:
            self.class_performance[class_id] = {'tp': 0, 'fp': 0, 'fn': 0}
    
    def _update_class_performance(self, predictions, batch):
        """Aktualisiert Performance-Metriken pro Klasse"""
        for pred, labels in zip(predictions, batch['mixed_labels']):
            pred_boxes = pred.boxes
            
            # True Positives und False Positives
            for box in pred_boxes:
                class_id = int(box.cls.item())
                matched = False
                
                for label in labels:
                    if label['class'] == class_id:
                        iou = self._calculate_iou(
                            box.xyxy[0].cpu(), 
                            torch.tensor(label['box'])
                        )
                        if iou > 0.5:
                            self.class_performance[class_id]['tp'] += 1
                            matched = True
                            break
                
                if not matched:
                    self.class_performance[class_id]['fp'] += 1
            
            # False Negatives
            for label in labels:
                class_id = label['class']
                matched = False
                
                for box in pred_boxes:
                    if int(box.cls.item()) == class_id:
                        iou = self._calculate_iou(
                            box.xyxy[0].cpu(), 
                            torch.tensor(label['box'])
                        )
                        if iou > 0.5:
                            matched = True
                            break
                
                if not matched:
                    self.class_performance[class_id]['fn'] += 1
    
    def _calculate_iou(self, box1, box2):
        """Berechnet IoU zwischen zwei Boxen"""
        # Intersection
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / (union + 1e-6)
    
    def _print_class_performance(self):
        """Gibt Performance-Metriken pro Klasse aus"""
        print("\nClass Performance:")
        for class_id, perf in self.class_performance.items():
            if perf['tp'] + perf['fp'] + perf['fn'] > 0:
                precision = perf['tp'] / (perf['tp'] + perf['fp']) if perf['tp'] + perf['fp'] > 0 else 0
                recall = perf['tp'] / (perf['tp'] + perf['fn']) if perf['tp'] + perf['fn'] > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                print(f"{TOOL_MAPPING[class_id]['name']}: "
                      f"F1={f1:.3f} (P={precision:.3f}, R={recall:.3f})")
    
    def _save_checkpoint(self, path):
        """Speichert Checkpoint mit allen relevanten Informationen"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Speichere Modell
        self.model.save(path / "model.pt")
        
        # Speichere Trainer-State
        torch.save({
            'class_weights': self.class_weights,
            'pseudo_label_threshold': self.pseudo_label_threshold,
            'loss_history': self.loss_history,
            'class_performance': self.class_performance
        }, path / "trainer_state.pt")
        
        # Speichere Performance-Metriken
        self._save_performance_metrics(path)
    
    def _save_performance_metrics(self, path):
        """Speichert detaillierte Performance-Metriken"""
        with open(path / "metrics.txt", "w") as f:
            f.write("Class Performance Details:\n\n")
            
            for class_id, perf in self.class_performance.items():
                if perf['tp'] + perf['fp'] + perf['fn'] > 0:
                    precision = perf['tp'] / (perf['tp'] + perf['fp']) if perf['tp'] + perf['fp'] > 0 else 0
                    recall = perf['tp'] / (perf['tp'] + perf['fn']) if perf['tp'] + perf['fn'] > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                    
                    f.write(f"\n{TOOL_MAPPING[class_id]['name']}:\n")
                    f.write(f"F1-Score: {f1:.3f}\n")
                    f.write(f"Precision: {precision:.3f}\n")
                    f.write(f"Recall: {recall:.3f}\n")
                    f.write(f"True Positives: {perf['tp']}\n")
                    f.write(f"False Positives: {perf['fp']}\n")
                    f.write(f"False Negatives: {perf['fn']}\n")
                    f.write(f"Current Weight: {self.class_weights[class_id]:.3f}\n")

def main():
    try:
        # Setup paths
        base_path = Path("/home/Bartscht/YOLO/surgical-instrument-action-detection")
        inference_weights = base_path / "models/hierarchical-surgical-workflow/Instrument-classification-detection/weights/instrument_detector/best_v35.pt"
        source_path = Path("/data/Bartscht/YOLO")
        target_path = Path("/data/Bartscht/HeiChole/domain_adaptation/train")
        save_dir = Path("/data/Bartscht/confmix_checkpoints")
        
        print("Initializing ConfMix training...")
        print(f"Source data: {source_path}")
        print(f"Target data: {target_path}")
        print(f"Save directory: {save_dir}")
        
        # Initialize and start training
        trainer = ConfMixTrainer(inference_weights)
        trainer.train(source_path, target_path, save_dir)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()