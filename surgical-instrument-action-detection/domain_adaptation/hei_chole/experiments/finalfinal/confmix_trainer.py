import torch
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from ultralytics import YOLO
from confmix_core import ConfidenceBasedDetector, ConfMixDetector
from dataset_loader import create_dataloaders, DetectionDataset
from torch.distributions import Normal
from PIL import Image
from torchvision import transforms

# Constants
CONFIDENCE_THRESHOLD = 0.1
IMAGE_SIZE = (640, 640)  # Standardgröße für alle Bilder

# Tool Mapping mit Performance-Gewichtung
TOOL_MAPPING = {
    0: {'name': 'grasper', 'weight': 0.3, 'base_threshold': 0.25},
    1: {'name': 'bipolar', 'weight': 1.0, 'base_threshold': 0.15},
    2: {'name': 'hook', 'weight': 0.3, 'base_threshold': 0.25},
    3: {'name': 'scissors', 'weight': 1.0, 'base_threshold': 0.15},
    4: {'name': 'clipper', 'weight': 1.0, 'base_threshold': 0.15},
    5: {'name': 'irrigator', 'weight': 1.0, 'base_threshold': 0.15},
    6: {'name': 'specimen_bag', 'weight': 0.1, 'base_threshold': 0.3}
}


"""
Refactored ConfMix implementation based on original paper approach
"""

class ConfMixTrainer:
    def __init__(self, model, device='cuda', save_dir=None):
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path('runs/train/confmix')
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.confidence_detector = ConfidenceBasedDetector(model)
        self.confmix_detector = ConfMixDetector(self.confidence_detector)
        
        # ConfMix specific parameters
        self.important_classes = [1, 3, 4, 5]  # bipolar, scissors, clipper, irrigator
        self.less_important_classes = [0, 2]   # grasper, hook
        
        # Confidence parameters
        self.class_uncertainty_params = {
            class_id: {
                'loc_std': 0.1 if info['weight'] > 0.5 else 0.2,
                'scale_std': 0.2 if info['weight'] > 0.5 else 0.3
            }
            for class_id, info in TOOL_MAPPING.items()
        }
        
        # Training settings
        self.num_epochs = 50
        self.batch_size = 8
        self.save_period = 5
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Performance tracking
        self.best_fitness = 0.0
        self.loss_history = []

    def _calculate_box_uncertainty(self, box, class_id):
        """Berechnet klassenspezifische Uncertainty für eine Bounding Box"""
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        params = self.class_uncertainty_params[class_id]
        
        # Klassenspezifische Standardabweichungen
        mu = torch.tensor([x1, y1, width, height])
        sigma = torch.tensor([
            width * params['loc_std'],   # Position X
            height * params['loc_std'],  # Position Y
            width * params['scale_std'], # Breite
            height * params['scale_std'] # Höhe
        ])
        
        distribution = Normal(mu, sigma)
        log_prob = distribution.log_prob(mu).mean()
        
        # Normalisierte Uncertainty
        uncertainty = 1 - torch.exp(log_prob).item()
        
        # Zusätzliche Gewichtung für wichtige Klassen
        if TOOL_MAPPING[class_id]['weight'] > 0.5:
            uncertainty *= 0.8  # Reduziere Uncertainty für wichtige Klassen
            
        return uncertainty
    
    def process_frame(self, target_image, progress_ratio):
        """Erweiterte ConfMix Region Selection Strategy"""
        if target_image.size != IMAGE_SIZE:
            target_image = target_image.resize(IMAGE_SIZE)
            
        regions = self._split_image_into_regions(target_image)
        detections = self.confidence_detector.predict_with_progressive_confidence(
            target_image, 
            progress_ratio
        )
        
        region_scores = self._calculate_region_scores(detections, regions, progress_ratio)
        
        # Adaptive Region Selection basierend auf Training Progress
        if progress_ratio < 0.3:
            # Frühe Phase: Fokus auf wichtige Klassen
            best_regions = self._select_regions_early_phase(region_scores)
        else:
            # Späte Phase: Ausgewogene Auswahl
            best_regions = self._select_regions_late_phase(region_scores)
            
        return {
            'regions': regions,
            'detections': detections,
            'region_scores': region_scores,
            'best_regions': best_regions,
            'progress_ratio': progress_ratio
        }
    
    def create_mixed_sample(self, target_image, source_image, target_results):
        """Erstellt gemischtes Sample mit Multi-Region Support und Masken"""
        if target_image.size != IMAGE_SIZE:
            target_image = target_image.resize(IMAGE_SIZE)
        if source_image.size != IMAGE_SIZE:
            source_image = source_image.resize(IMAGE_SIZE)
                
        # Erstelle Basis-Maske für Feature Consistency
        mask = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]))
        
        # Fülle ausgewählte Regionen
        selected_regions = []
        for region_idx in target_results['best_regions']:
            x1, y1, x2, y2 = target_results['regions'][region_idx]
            mask[y1:y2, x1:x2] = 1
            selected_regions.append(target_results['regions'][region_idx])
        
        # Mix images
        target_array = np.array(target_image)
        source_array = np.array(source_image)
        mixed_array = source_array.copy()
        
        # Ersetze ausgewählte Regionen
        for x1, y1, x2, y2 in selected_regions:
            mixed_array[y1:y2, x1:x2] = target_array[y1:y2, x1:x2]
        
        # Erstelle source detections mit dem inference model
        source_tensor = transforms.ToTensor()(source_image).unsqueeze(0)
        with torch.no_grad():
            source_detections = self.confidence_detector.model(source_tensor)[0]
        
        # Filtere und kombiniere Detektionen
        mixed_detections = self._combine_detections(
            target_results['detections'],
            selected_regions,
            source_detections.boxes.data.cpu().numpy(), 
            self.important_classes
        )
        
        return {
            'mixed_image': Image.fromarray(mixed_array.astype('uint8')),
            'mixing_mask': mask,
            'selected_regions': selected_regions,
            'mixed_detections': mixed_detections,
            'source_detections': source_detections,
            'target_detections': target_results['detections'],  
            'confidence': np.mean([det['combined_confidence'] 
                                for det in mixed_detections]) if mixed_detections else 0
        }
    def train_step(self, source_batch, target_batch, progress_ratio):
        source_images = source_batch['images'].to(self.device)
        target_images = target_batch['images'].to(self.device)
        
        # Resize-Sicherheit für das Modell
        if source_images.shape[-2:] != (640, 640):
            source_images = torch.nn.functional.interpolate(
                source_images, 
                size=(640, 640), 
                mode='bilinear', 
                align_corners=False
            )
        
        if target_images.shape[-2:] != (640, 640):
            target_images = torch.nn.functional.interpolate(
                target_images, 
                size=(640, 640), 
                mode='bilinear', 
                align_corners=False
            )

        # Debug: Dimensionen der Ursprungsbilder
        print("Source images shape:", source_images.shape)
        print("Target images shape:", target_images.shape)

        # 1. Pseudo-Labels generieren
        with torch.no_grad():
            target_results = self.confmix_detector.process_frame(target_images, progress_ratio)
        
        # 2. ConfMix Sample erstellen
        mixed_data = self.confmix_detector.create_mixed_sample(
            target_images,
            source_images,
            target_results
        )
        # Debug: Detaillierte Ausgabe der mixed_data
        print("\n--- Mixed Data Debug ---")
        for key, value in mixed_data.items():
            if torch.is_tensor(value):
                print(f"{key} shape: {value.shape}")
            else:
                print(f"{key} type: {type(value)}")

        print("Source images shape:", source_images.shape)
        print("Target images shape:", target_images.shape)
        print("Source images dtype:", source_images.dtype)
        print("Target images dtype:", target_images.dtype)
        
        mixed_image = mixed_data['mixed_image']
        print("\nMixed Image Details:")
        print("Type:", type(mixed_image))
        print("Is Tensor:", torch.is_tensor(mixed_image))
        if torch.is_tensor(mixed_image):
            print("Shape:", mixed_image.shape)
            print("Dimensions:", mixed_image.ndimension())

        # 3. Forward pass mit gemischtem Bild
        mixed_predictions = self.model(mixed_data['mixed_image'])
        
        # 4. Detection Loss auf Source-Region
        det_loss = self.compute_detection_loss(
            mixed_predictions, 
            source_batch['labels'],
            mixed_data['mixing_mask']
        )
        
        # 5. Consistency Loss auf Target-Region
        consistency_loss = self.compute_consistency_loss(
            mixed_predictions,
            mixed_data['mixed_detections'],
            mixed_data['mixing_mask']
        )
        
        # 6. Kombinierter Loss mit Gewichtung
        gamma = mixed_data['confidence']  # Confidence-basierte Gewichtung
        total_loss = det_loss + gamma * consistency_loss
        
        # 7. Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def compute_detection_loss(self, predictions, labels, mask):
        """Detection Loss mit Klassengewichtung"""
        loss = 0
        for pred, label in zip(predictions, labels):
            class_id = label['class']
            class_weight = TOOL_MAPPING[class_id]['weight']
            
            # Nur auf Source-Region anwenden
            masked_pred = pred * (1 - mask)
            class_loss = self.model.criterion(masked_pred, label)
            loss += class_weight * class_loss
        return loss

    def compute_consistency_loss(self, predictions, pseudo_labels, mask):
        """Consistency Loss für Pseudo-Labels"""
        if not pseudo_labels:
            return torch.tensor(0.0, device=self.device)
            
        loss = 0
        for pred, label in zip(predictions, pseudo_labels):
            class_id = label['class']
            
            # Höhere Gewichtung für wichtige Klassen
            class_weight = 1.0 if class_id in self.important_classes else 0.5
            
            # Nur auf Target-Region anwenden
            masked_pred = pred * mask
            class_loss = self.model.criterion(masked_pred, label)
            loss += class_weight * class_loss
        return loss
    
    def _update_class_performance(self, predictions, ground_truth):
        """Performance Tracking pro Klasse"""
        for cls_id in TOOL_MAPPING.keys():
            pred_count = sum(1 for p in predictions if p['class'] == cls_id)
            gt_count = sum(1 for g in ground_truth if g['class'] == cls_id)
            
            self.class_performance[cls_id]['tp'] += min(pred_count, gt_count)
            self.class_performance[cls_id]['fp'] += max(0, pred_count - gt_count)
            self.class_performance[cls_id]['fn'] += max(0, gt_count - pred_count)