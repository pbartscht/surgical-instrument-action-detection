import torch
from pathlib import Path
import numpy as np
from base_setup import DualModelManager, TOOL_MAPPING
from confmix_core import ConfidenceBasedDetector, ConfMixDetector
from dataset_loader import create_confmix_dataloader



class FeatureMemoryBank:
    def __init__(self, size=1000):
        self.features = []
        self.max_size = size
    
    def update(self, new_features):
        self.features.extend(new_features)
        if len(self.features) > self.max_size:
            self.features = self.features[-self.max_size:]
    
    def get_features(self):
        return self.features
    


class ConfMixTrainer:
    def __init__(self, inference_weights, train_weights=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.dual_model_manager = DualModelManager(inference_weights, train_weights)
        self.confidence_detector = ConfidenceBasedDetector(
            self.dual_model_manager.inference_model
        )
        self.confmix_detector = ConfMixDetector(self.confidence_detector)
        
        self.feature_bank = FeatureMemoryBank()
        self.total_steps = 0

        # Training settings bleiben gleich
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
    
    def _train_source_batch(self, batch):
        """Trainiert auf Source-Daten mit Klassengewichtung und erweiterten Features"""
        def custom_source_loss(pred, batch_data):
            loss = torch.tensor(0.0, device=self.device)
            
            # Extrahiere Vorhersagen und Ground Truth
            for pred_boxes, target_boxes in zip(pred, batch_data['labels']):
                if not target_boxes:
                    continue
                    
                # Berechne IoU Matrix zwischen Vorhersagen und Ground Truth
                ious = self._calculate_iou_matrix(pred_boxes, target_boxes)
                
                # Matchmaking zwischen Vorhersagen und Ground Truth
                matches = self._match_boxes(ious, threshold=0.5)
                
                for pred_idx, target_idx in matches:
                    pred_box = pred_boxes[pred_idx]
                    target_box = target_boxes[target_idx]
                    
                    # Klassengewichtung
                    class_id = target_box['class']
                    class_weight = self.class_weights[class_id]
                    
                    # Regression Loss (CIoU)
                    box_loss = self._calculate_ciou_loss(
                        pred_box['boxes'],
                        torch.tensor(target_box['box']).to(self.device)
                    )
                    
                    # Classification Loss
                    cls_loss = torch.nn.functional.cross_entropy(
                        pred_box['cls'],
                        torch.tensor([class_id]).to(self.device)
                    )
                    
                    # Gewichteter Gesamtverlust
                    loss += class_weight * (box_loss + cls_loss)
            
            # Normalisierung
            num_targets = sum(len(t) for t in batch_data['labels'])
            if num_targets > 0:
                loss = loss / num_targets
                
            return loss

        try:
            # Prepare batch in YOLO format
            batch_dict = {
                'images': batch['source_images'],
                'labels': batch['source_labels']
            }
            
            # Training durchführen
            results = self.dual_model_manager.train_adapter.train_step(
                batch_dict, 
                custom_loss_fn=custom_source_loss
            )
            
            # Update Counter
            self.dual_model_manager.update_counter += 1
            
            return results['loss']
            
        except Exception as e:
            print(f"Error in source batch training: {str(e)}")
            return torch.tensor(0.0, device=self.device)
        
    def _train_mixed_batch(self, batch):
        """Trainiert auf gemischten Samples mit Feature Memory und Domain Alignment"""
        def custom_mixed_loss(results, batch_data):
            loss = torch.tensor(0.0, device=self.device)
            for idx, result in enumerate(results):
                boxes = result.boxes
                labels = batch_data['labels'][idx]
                
                for box, label in zip(boxes, labels):
                    class_id = int(label['class'])
                    # Erhöhte Gewichtung für wichtige Klassen
                    class_weight = self.class_weights[class_id]
                    if class_id in self.confmix_detector.important_classes:
                        class_weight *= 1.5
                    
                    # Box loss mit Koordinaten
                    box_loss = torch.nn.functional.mse_loss(
                        box.xyxy,
                        torch.tensor(label['box']).to(box.xyxy.device)
                    )
                    loss += class_weight * box_loss
            
            return loss / len(batch_data['data'])

        # Prepare mixed batch in YOLO format
        batch_dict = {
            'data': batch['mixed_images'],
            'batch_idx': list(range(len(batch['mixed_images']))),
            'im_file': [''] * len(batch['mixed_images']),
            'labels': [
                [{
                    'cls': [label['class']],
                    'bboxes': label['box'],
                    'segments': [],
                    'keypoints': None,
                    'normalized': True,
                    'bbox_format': 'xywh'
                } for label in img_labels]
                for img_labels in batch['mixed_labels']
            ]
        }

        try:
            # Feature Extraction für Domain Alignment
            with torch.no_grad():
                mixed_features = self.dual_model_manager.inference_model.model.backbone(
                    batch['mixed_images']
                )
                self.feature_bank.update(mixed_features.detach())

            # Target Domain Alignment Loss
            target_features = self.feature_bank.get_features()
            if target_features:
                target_features = torch.stack(target_features[:len(mixed_features)])
                alignment_loss = torch.nn.functional.mse_loss(
                    mixed_features,
                    target_features
                )
            else:
                alignment_loss = torch.tensor(0.0, device=self.device)

            # Training Step mit kombiniertem Loss
            mixed_loss = self.dual_model_manager.train_step(batch_dict, custom_mixed_loss)
            if isinstance(mixed_loss, dict):
                mixed_loss = mixed_loss.get('loss', 0.0)
            
            # Kombiniere Mixed Loss und Alignment Loss
            total_loss = mixed_loss + 0.1 * alignment_loss
            
            self.dual_model_manager.update_counter += 1
            return total_loss

        except Exception as e:
            print(f"Error in mixed batch training: {str(e)}")
            return torch.tensor(0.0, device=self.device)
    
    def _calculate_consistency_loss(self, mixed_images, source_images, mixing_masks, mixed_results, source_results):
        """Berechnet Consistency Loss zwischen Mixed und Source Predictions"""
        consistency_loss = 0
        
        for mixed_pred, source_pred, mask in zip(mixed_results, source_results, mixing_masks):
            # Nur für Source Regionen (mask == 0)
            source_regions = ~mask.bool()
            
            # Finde überlappende Detektionen
            for mixed_box in mixed_pred.boxes:
                box_center = self._get_box_center(mixed_box)
                if not mask[int(box_center[1]), int(box_center[0])]:  # In Source Region
                    # Finde matching box in source predictions
                    best_match = self._find_best_matching_box(mixed_box, source_pred.boxes)
                    if best_match is not None:
                        # Berechne Consistency Loss
                        box_loss = self._compute_box_consistency(mixed_box, best_match)
                        class_id = int(mixed_box.cls.item())
                        # Gewichte Loss basierend auf Klassengewichten
                        consistency_loss += self.class_weights[class_id] * box_loss
        
        return consistency_loss / len(mixed_images)
    
    def _compute_pred_consistency(self, mixed_result, source_result, mask):
        """Berechnet Consistency zwischen Predictions basierend auf Mixing Mask"""
        mask = mask.bool()
        loss = 0
        
        # Extrahiere Predictions in nicht-gemischten Regionen
        mixed_boxes = mixed_result.boxes
        source_boxes = source_result.boxes
        
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
        results = self.dual_model_manager.inference_model(batch['mixed_images'])
        
        for result, labels in zip(results, batch['mixed_labels']):
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls.item())
                # Prüfe ob true positive oder false positive
                matched = False
                for label in labels:
                    if label['class'] == class_id:
                        if self._calculate_iou(box.xyxy[0], label['box']) > 0.5:
                            self.class_performance[class_id]['tp'] += 1
                            matched = True
                            break
                if not matched:
                    self.class_performance[class_id]['fp'] += 1
            
            # Zähle false negatives
            for label in labels:
                class_id = label['class']
                matched = False
                for box in boxes:
                    if int(box.cls.item()) == class_id:
                        if self._calculate_iou(box.xyxy[0], label['box']) > 0.5:
                            matched = True
                            break
                if not matched:
                    self.class_performance[class_id]['fn'] += 1
    
    def _calculate_iou(self, box1, box2):
        """Berechnet IoU zwischen zwei Boxen"""
        # Konvertiere zu Tensoren wenn nötig
        if not isinstance(box1, torch.Tensor):
            box1 = torch.tensor(box1)
        if not isinstance(box2, torch.Tensor):
            box2 = torch.tensor(box2)
            
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        
        return intersection / (union + 1e-6)  # Kleine Konstante zur numerischen Stabilität
    
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
        """Speichert Performance-Metriken mit detaillierter Analyse"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / 'performance.txt', 'w') as f:
            # Gesamtperformance
            total_tp = sum(perf['tp'] for perf in self.class_performance.values())
            total_fp = sum(perf['fp'] for perf in self.class_performance.values())
            total_fn = sum(perf['fn'] for perf in self.class_performance.values())
            
            overall_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
            overall_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
            overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if overall_precision + overall_recall > 0 else 0
            
            f.write(f"Overall Performance:\n")
            f.write(f"F1={overall_f1:.3f} (P={overall_precision:.3f}, R={overall_recall:.3f})\n\n")
            
            # Performance pro Klasse
            f.write("Per-Class Performance:\n")
            for class_id, perf in self.class_performance.items():
                if perf['tp'] + perf['fp'] + perf['fn'] > 0:
                    precision = perf['tp'] / (perf['tp'] + perf['fp']) if perf['tp'] + perf['fp'] > 0 else 0
                    recall = perf['tp'] / (perf['tp'] + perf['fn']) if perf['tp'] + perf['fn'] > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                    
                    # Erweiterte Metriken für wichtige Klassen
                    class_weight = TOOL_MAPPING[class_id]['weight']
                    is_important = class_weight > 0.5
                    
                    f.write(f"\n{TOOL_MAPPING[class_id]['name']}:\n")
                    f.write(f"- F1={f1:.3f} (P={precision:.3f}, R={recall:.3f})\n")
                    f.write(f"- Class Weight: {class_weight}\n")
                    f.write(f"- Total Detections: {perf['tp'] + perf['fp']}\n")
                    f.write(f"- True Positives: {perf['tp']}\n")
                    f.write(f"- False Positives: {perf['fp']}\n")
                    f.write(f"- False Negatives: {perf['fn']}\n")
                    
                    if is_important:
                        # Zusätzliche Analyse für wichtige Klassen
                        miss_rate = perf['fn'] / (perf['tp'] + perf['fn']) if perf['tp'] + perf['fn'] > 0 else 1
                        false_discovery_rate = perf['fp'] / (perf['tp'] + perf['fp']) if perf['tp'] + perf['fp'] > 0 else 1
                        
                        f.write(f"- Miss Rate: {miss_rate:.3f}\n")
                        f.write(f"- False Discovery Rate: {false_discovery_rate:.3f}\n")
                        
                        # Performanzwarnung wenn nötig
                        if f1 < 0.3:
                            f.write("WARNING: Low performance on important class!\n")

    def train(self, source_path, target_path, save_dir):
        """Haupttrainingsschleife"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Starting training with source: {source_path}, target: {target_path}")
        
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
            # Progress ratio für adaptive Gewichtung
            progress_ratio = batch['progress_ratio'].mean()
            
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Generate predictions for mixed and source images
            with torch.no_grad():
                mixed_results = self.dual_model_manager.inference_model(
                    batch['mixed_images']
                )
                source_results = self.dual_model_manager.inference_model(
                    batch['source_images']
                )
            
            # Erweiterte Loss-Berechnung
            # 1. Source Training
            source_loss = self._train_source_batch(batch) * max(0.1, 1.0 - progress_ratio)
            
            # 2. Mixed Training
            mixed_loss = self._train_mixed_batch(batch)
            
            # 3. Consistency Loss
            consistency_loss = self._calculate_consistency_loss(
                batch['mixed_images'],
                batch['source_images'],
                batch['mixing_masks'],
                mixed_results,
                source_results
            ) * min(1.0, progress_ratio * 2)
            
            # Gesamtverlust
            total_loss = source_loss + mixed_loss + consistency_loss
            
            epoch_losses.append(float(total_loss))
            
            # Update Modell mit kontrolliertem Target-Fokus
            if self.dual_model_manager.update_counter >= self.dual_model_manager.update_frequency:
                self._update_model_weights()  # Neue Methode
                self.confidence_detector.model = self.dual_model_manager.inference_model
            
            # Progress & Performance Tracking
            if (batch_idx + 1) % 10 == 0:
                self._update_class_performance(batch)
                print(f"Batch {batch_idx+1}/{len(dataloader)} - "
                    f"Loss: {total_loss:.4f}")
        
        return np.mean(epoch_losses)

    def _update_model_weights(self):
        """Kontrollierter Update der Modell-Gewichte"""
        progress = self.dual_model_manager.update_counter / self.total_steps
        
        if progress < 0.2:
            # Frühe Phase: Voller Update
            self.dual_model_manager._update_inference_model()
        elif progress < 0.5:
            # Mittlere Phase: Selektiver Update
            self._update_shared_layers()
        else:
            # Späte Phase: Target-fokussiert
            self._update_target_layers()
            
    def _update_shared_layers(self):
        """Update nur für shared layers"""
        source_state = self.dual_model_manager.train_model.model.state_dict()
        target_state = self.dual_model_manager.inference_model.model.state_dict()
        
        # Update nur backbone/shared layers
        for name, param in source_state.items():
            if 'backbone' in name or 'shared' in name:
                target_state[name].copy_(param)
                
    def _update_target_layers(self):
        """Update nur für target-spezifische Layer"""
        source_state = self.dual_model_manager.train_model.model.state_dict()
        target_state = self.dual_model_manager.inference_model.model.state_dict()
        
        # Update nur detection/classifier layers
        for name, param in source_state.items():
            if 'detect' in name or 'classifier' in name:
                target_state[name].copy_(param)
                
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
        
        # Initialize trainer
        trainer = ConfMixTrainer(inference_weights)
        
        # Start training
        trainer.train(source_path, target_path, save_dir)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()