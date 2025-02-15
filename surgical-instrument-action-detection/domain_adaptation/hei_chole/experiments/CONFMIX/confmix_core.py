import math
import numpy as np
from PIL import Image
import torch
from torch.distributions import Normal
from base_setup import TOOL_MAPPING, IMAGE_SIZE
from torchvision import transforms

class ConfidenceBasedDetector:
    def __init__(self, model, confidence_threshold=0.25):
        self.model = model
        self.confidence_threshold = confidence_threshold
        # Klassenspezifische Uncertainty-Parameter
        self.class_uncertainty_params = {
            class_id: {
                'loc_std': 0.1 if info['weight'] > 0.5 else 0.2,  # Konservativer für wichtige Klassen
                'scale_std': 0.2 if info['weight'] > 0.5 else 0.3
            }
            for class_id, info in TOOL_MAPPING.items()
        }
    
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

    def predict_with_progressive_confidence(self, image, progress_ratio):
        """Erweiterte Progressive Confidence-Berechnung mit klassenspezifischer Anpassung"""
        results = self.model(image)
        enhanced_detections = []
        
        # Dynamische Confidence-Anpassung
        alpha = 5.0  # Steuerung der Progressionsgeschwindigkeit
        delta = 2 / (1 + math.exp(-alpha * progress_ratio)) - 1
        
        for detection in results[0].boxes:
            class_id = int(detection.cls.item())
            
            # Überspringe specimen_bag class
            if class_id == 6:
                continue
                
            detector_confidence = detection.conf.item()
            box = detection.xyxy[0].cpu().numpy()
            
            # Klassenspezifische Box Uncertainty
            box_uncertainty = self._calculate_box_uncertainty(box, class_id)
            box_confidence = 1 - box_uncertainty
            
            # Adaptive Schwellenwerte für verschiedene Klassen
            base_threshold = TOOL_MAPPING[class_id]['base_threshold']
            weight = TOOL_MAPPING[class_id]['weight']
            
            # Niedrigerer Anfangsschwellwert für wichtige Klassen
            if weight > 0.5:
                class_threshold = base_threshold * (0.5 + 0.5 * progress_ratio)
            else:
                class_threshold = base_threshold
            
            # Kombinierte und gewichtete Confidence
            combined_confidence = (
                (1 - delta) * detector_confidence + 
                delta * (detector_confidence * box_confidence)
            ) * weight
            
            if combined_confidence >= class_threshold:
                enhanced_detections.append({
                    'box': box,
                    'class': class_id,
                    'detector_confidence': detector_confidence,
                    'box_confidence': box_confidence,
                    'combined_confidence': combined_confidence,
                    'instrument_name': TOOL_MAPPING[class_id]['name']
                })
        
        return enhanced_detections

class ConfMixDetector:
    def __init__(self, confidence_detector):
        self.confidence_detector = confidence_detector
        # Definiere wichtige und weniger wichtige Klassen
        self.important_classes = [1, 3, 4, 5]  # bipolar, scissors, clipper, irrigator
        self.less_important_classes = [0, 2]   # grasper, hook
        
        # Gewichte für Region-Scoring
        self.region_weights = {
            'confidence': 0.4,
            'important_class': 0.4,
            'diversity': 0.2
        }
    
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
    
    def _split_image_into_regions(self, image):
        """Teilt Bild in 4 gleiche Regionen"""
        width, height = image.size
        regions = [
            (0, 0, width//2, height//2),
            (width//2, 0, width, height//2),
            (0, height//2, width//2, height),
            (width//2, height//2, width, height)
        ]
        return regions
    
    def _calculate_region_scores(self, detections, regions, progress_ratio):
        """Berechnet erweiterte Region-Scores mit Klassenwichtung"""
        region_scores = []
        
        for region in regions:
            region_dets = self._get_detections_in_region(detections, region)
            
            # Basis-Score aus Confidence
            confidence_score = np.mean([det['combined_confidence'] 
                                     for det in region_dets]) if region_dets else 0
            
            # Score für wichtige Klassen
            important_class_score = self._calculate_important_class_score(region_dets)
            
            # Diversitäts-Score
            diversity_score = self._calculate_diversity_score(region_dets)
            
            # Gewichteter Gesamt-Score
            total_score = (
                self.region_weights['confidence'] * confidence_score +
                self.region_weights['important_class'] * important_class_score +
                self.region_weights['diversity'] * diversity_score
            )
            
            region_scores.append({
                'detections': region_dets,
                'confidence_score': confidence_score,
                'important_class_score': important_class_score,
                'diversity_score': diversity_score,
                'total_score': total_score
            })
            
        return region_scores

    def _calculate_important_class_score(self, detections):
        """Berechnet Score basierend auf wichtigen Klassen"""
        if not detections:
            return 0
            
        important_count = sum(1 for det in detections 
                            if det['class'] in self.important_classes)
        return important_count / len(detections)
    
    def _calculate_diversity_score(self, detections):
        """Berechnet Diversitäts-Score basierend auf Klassenverteilung"""
        if not detections:
            return 0
            
        unique_classes = len(set(det['class'] for det in detections))
        return unique_classes / len(TOOL_MAPPING)
    
    def _select_regions_early_phase(self, region_scores, top_k=2):
        """Wählt Regionen mit Fokus auf wichtige Klassen"""
        sorted_regions = sorted(
            range(len(region_scores)),
            key=lambda i: (
                region_scores[i]['important_class_score'],
                region_scores[i]['confidence_score']
            ),
            reverse=True
        )
        return sorted_regions[:top_k]
    
    def _select_regions_late_phase(self, region_scores, top_k=2):
        """Wählt Regionen basierend auf Gesamt-Score"""
        sorted_regions = sorted(
            range(len(region_scores)),
            key=lambda i: region_scores[i]['total_score'],
            reverse=True
        )
        return sorted_regions[:top_k]
    
    def _get_detections_in_region(self, detections, region):
        """Filtert Detektionen für eine Region"""
        x1, y1, x2, y2 = region
        region_dets = []
        
        for det in detections:
            box = det['box']
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            
            if (x1 <= center_x <= x2 and y1 <= center_y <= y2):
                region_dets.append(det)
                
        return region_dets

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

    def _combine_detections(self, target_dets, selected_regions, source_dets, important_classes):
        """Kombiniert source und target detections basierend auf Regionen"""
        combined_dets = []
        
        # Konvertiere Regionen in eine Maske für schnelleres Matching
        region_mask = np.zeros(IMAGE_SIZE, dtype=bool)
        for x1, y1, x2, y2 in selected_regions:
            region_mask[y1:y2, x1:x2] = True
        
        # Verarbeite target detections
        for det in target_dets:
            box = det['box']
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            
            # Prüfe ob detection im target bereich liegt
            if region_mask[int(center_y), int(center_x)]:
                combined_dets.append(det)
        
        # Verarbeite source detections
        for det in source_dets:
            box = det[:4]  # YOLO format [x1, y1, x2, y2]
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            
            # Nur hinzufügen wenn außerhalb der target regionen
            if not region_mask[int(center_y), int(center_x)]:
                combined_dets.append({
                    'box': box,
                    'class': int(det[5]),  # YOLO class index
                    'detector_confidence': det[4],  # YOLO confidence
                    'box_confidence': 1.0,  # Default für source
                    'combined_confidence': det[4],
                    'instrument_name': TOOL_MAPPING[int(det[5])]['name']
                })
        
        return combined_dets