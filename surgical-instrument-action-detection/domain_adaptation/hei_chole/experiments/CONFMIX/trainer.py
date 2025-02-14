import torch
from pathlib import Path
import numpy as np
from base_setup import DualModelManager, TOOL_MAPPING
from confmix_core import ConfidenceBasedDetector, ConfMixDetector
from dataset_loader import create_confmix_dataloader

class ConfMixTrainer:
    def __init__(self, inference_weights, train_weights=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.dual_model_manager = DualModelManager(inference_weights, train_weights)
        self.confidence_detector = ConfidenceBasedDetector(
            self.dual_model_manager.inference_model
        )
        self.confmix_detector = ConfMixDetector(self.confidence_detector)
        
        # Training settings
        self.num_epochs = 50
        self.batch_size = 8
        self.num_workers = 0
        self.save_frequency = 5
        
        # Class weights für Loss
        self.class_weights = self._initialize_class_weights()
        
        # Tracking der Performance
        self.class_performance = {
            class_id: {'tp': 0, 'fp': 0, 'fn': 0} 
            for class_id in TOOL_MAPPING.keys()
        }
        
    def train(self, source_path, target_path, save_dir):
        """Haupttrainingsschleife"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup DataLoader
        dataloader = create_confmix_dataloader(
            source_path, target_path, self.confmix_detector,
            self.batch_size, self.num_workers
        )
        
        # Training Loop
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            epoch_loss = self._train_epoch(dataloader)
            
            # Update class weights basierend auf Performance
            self._update_class_weights()
            
            print(f"Epoch {epoch+1} - Average Loss: {epoch_loss:.4f}")
            self._print_class_performance()
            
            # Periodisches Speichern
            if (epoch + 1) % self.save_frequency == 0:
                checkpoint_dir = save_dir / f"epoch_{epoch+1}"
                self.dual_model_manager.save_models(checkpoint_dir)
                self._save_class_performance(checkpoint_dir)
                print(f"Saved checkpoint to {checkpoint_dir}")
    
    def _train_epoch(self, dataloader):
        """Trainiert eine Epoche mit erweitertem Loss"""
        epoch_losses = []
        self._reset_class_performance()
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # 1. Source Training mit Class Weights
            source_loss = self._train_source_batch({
                'imgs': batch['source_images'],
                'labels': batch['source_labels']
            })
            
            # 2. Mixed Sample Training
            mixed_loss = self._train_mixed_batch({
                'imgs': batch['mixed_images'],
                'labels': batch['mixed_labels']
            })
            
            # 3. Consistency Loss
            consistency_loss = self._calculate_consistency_loss(
                batch['mixed_images'],
                batch['source_images'],
                batch['mixing_masks'],
                batch['progress_ratio']
            )
            
            # Gewichteter Gesamtverlust
            total_loss = (
                source_loss + 
                mixed_loss + 
                self._get_consistency_weight(batch['progress_ratio']) * consistency_loss
            )
            
            epoch_losses.append(float(total_loss))
            
            # Update Inference Model wenn nötig
            if self.dual_model_manager.update_counter >= self.dual_model_manager.update_frequency:
                self.dual_model_manager._update_inference_model()
                # Update ConfMix Detector
                self.confidence_detector.model = self.dual_model_manager.inference_model
            
            # Progress & Performance Tracking
            if (batch_idx + 1) % 10 == 0:
                self._update_class_performance(batch)
                print(f"Batch {batch_idx+1}/{len(dataloader)} - "
                      f"Loss: {total_loss:.4f}")
        
        return np.mean(epoch_losses)
    
    def _train_source_batch(self, batch):
        """Trainiert auf Source-Daten mit Klassengewichtung"""
        predictions = self.dual_model_manager.train_model(batch['imgs'])
        loss = 0
        
        for pred, labels in zip(predictions, batch['labels']):
            for label in labels:
                class_id = label['class']
                class_weight = self.class_weights[class_id]
                loss += class_weight * self._calculate_detection_loss(pred, label)
                
        return loss / len(batch['imgs'])
    
    def _train_mixed_batch(self, batch):
        """Trainiert auf gemischten Samples"""
        return self.dual_model_manager.train_step({
            'imgs': batch['imgs'],
            'labels': batch['labels']
        })
    
    def _calculate_consistency_loss(self, mixed_images, source_images, mixing_masks, progress_ratio):
        """Berechnet Consistency Loss zwischen Mixed und Original Predictions"""
        # Predictions auf Mixed Images
        mixed_preds = self.dual_model_manager.predict_with_inference_model(mixed_images)
        
        # Predictions auf Source Images
        source_preds = self.dual_model_manager.predict_with_inference_model(source_images)
        
        loss = 0
        for mixed_pred, source_pred, mask, ratio in zip(mixed_preds, source_preds, 
                                                      mixing_masks, progress_ratio):
            pred_loss = self._compute_pred_consistency(
                mixed_pred, source_pred, mask
            )
            # Gewichte Loss mit Progress Ratio
            loss += pred_loss * ratio
        
        return loss / len(mixed_images)
    
    def _compute_pred_consistency(self, mixed_pred, source_pred, mask):
        """Berechnet Consistency zwischen Predictions basierend auf Mixing Mask"""
        mask = mask.bool()
        loss = 0
        
        # Extrahiere Predictions in nicht-gemischten Regionen
        mixed_boxes = mixed_pred[0].boxes
        source_boxes = source_pred[0].boxes
        
        for box in mixed_boxes:
            box_center = ((box.xyxy[0][0] + box.xyxy[0][2])/2, 
                         (box.xyxy[0][1] + box.xyxy[0][3])/2)
            
            # Nur für Regionen außerhalb der Mixing-Maske
            if not mask[int(box_center[1]), int(box_center[0])]:
                closest_source_box = self._find_closest_box(box, source_boxes)
                if closest_source_box is not None:
                    # Gewichte Loss mit Klassengewicht
                    class_id = int(box.cls.item())
                    class_weight = self.class_weights[class_id]
                    loss += class_weight * torch.nn.functional.mse_loss(
                        box.xyxy, closest_source_box.xyxy
                    )
        
        return loss
    
    def _get_consistency_weight(self, progress_ratio):
        """Berechnet Gewicht für Consistency Loss"""
        # Starte mit niedrigem Gewicht, erhöhe während Training
        return min(1.0, 0.2 + 0.8 * progress_ratio)
    
    def _initialize_class_weights(self):
        """Initialisiert Klassengewichte basierend auf TOOL_MAPPING"""
        return {class_id: info['weight'] for class_id, info in TOOL_MAPPING.items()}
    
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
    
    def _update_class_performance(self, batch):
        """Aktualisiert Performance-Metriken pro Klasse"""
        predictions = self.dual_model_manager.inference_model(batch['mixed_images'])
        
        for pred, labels in zip(predictions, batch['mixed_labels']):
            pred_boxes = pred[0].boxes
            for box in pred_boxes:
                class_id = int(box.cls.item())
                # Prüfe ob true positive oder false positive
                matched = False
                for label in labels:
                    if label['class'] == class_id:
                        if self._calculate_iou(box.xyxy, label['box']) > 0.5:
                            self.class_performance[class_id]['tp'] += 1
                            matched = True
                            break
                if not matched:
                    self.class_performance[class_id]['fp'] += 1
            
            # Zähle false negatives
            for label in labels:
                class_id = label['class']
                matched = False
                for box in pred_boxes:
                    if int(box.cls.item()) == class_id:
                        if self._calculate_iou(box.xyxy, label['box']) > 0.5:
                            matched = True
                            break
                if not matched:
                    self.class_performance[class_id]['fn'] += 1
    
    def _calculate_iou(self, box1, box2):
        """Berechnet IoU zwischen zwei Boxen"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (box1_area + box2_area - intersection)
    
    def _find_closest_box(self, ref_box, boxes):
        """Findet die nächstgelegene Box aus einer Liste von Boxes"""
        if len(boxes) == 0:
            return None
            
        ref_center = ((ref_box.xyxy[0][0] + ref_box.xyxy[0][2])/2, 
                     (ref_box.xyxy[0][1] + ref_box.xyxy[0][3])/2)
        
        min_dist = float('inf')
        closest_box = None
        
        for box in boxes:
            center = ((box.xyxy[0][0] + box.xyxy[0][2])/2, 
                     (box.xyxy[0][1] + box.xyxy[0][3])/2)
            dist = ((ref_center[0] - center[0])**2 + 
                   (ref_center[1] - center[1])**2).sqrt()
            
            if dist < min_dist:
                min_dist = dist
                closest_box = box
        
        return closest_box
    
    def _print_class_performance(self):
        """Gibt Performance-Metriken pro Klasse aus"""
        print("\nClass Performance:")
        for class_id, perf in self.class_performance.items():
            if perf['tp'] + perf['fp'] + perf['fn'] > 0:
                precision = perf['tp'] / (perf['tp'] + perf['fp']) if perf['tp'] + perf['fp'] > 0 else 0
                recall = perf['tp'] / (perf['tp'] + perf['fn']) if perf['tp'] + perf['fn'] > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                print(f"{TOOL_MAPPING[class_id]['name']}: F1={f1:.3f} (P={precision:.3f}, R={recall:.3f})")
    
    def _save_class_performance(self, save_dir):
        """Speichert Performance-Metriken"""
        save_dir = Path(save_dir)
        with open(save_dir / 'performance.txt', 'w') as f:
            for class_id, perf in self.class_performance.items():
                if perf['tp'] + perf['fp'] + perf['fn'] > 0:
                    precision = perf['tp'] / (perf['tp'] + perf['fp']) if perf['tp'] + perf['fp'] > 0 else 0
                    recall = perf['tp'] / (perf['tp'] + perf['fn']) if perf['tp'] + perf['fn'] > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                    f.write(f"{TOOL_MAPPING[class_id]['name']}: ")
                    f.write(f"F1={f1:.3f} (P={precision:.3f}, R={recall:.3f})\n")

def main():
    # Setup paths
    base_path = Path("/home/Bartscht/YOLO/surgical-instrument-action-detection")
    inference_weights = base_path / "models/hierarchical-surgical-workflow/Instrument-classification-detection/weights/instrument_detector/best_v35.pt"
    source_path = Path("/data/Bartscht/YOLO")
    target_path = Path("/data/Bartscht/HeiChole/domain_adaptation/train")
    save_dir = Path("/data/Bartscht/confmix_checkpoints")
    
    # Initialize trainer
    trainer = ConfMixTrainer(inference_weights)
    
    # Start training
    trainer.train(source_path, target_path, save_dir)

if __name__ == "__main__":
    main()