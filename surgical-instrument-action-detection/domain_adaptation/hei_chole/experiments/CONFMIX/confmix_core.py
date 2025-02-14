import math
import numpy as np
from PIL import Image
import torch
from base_setup import TOOL_MAPPING, IMAGE_SIZE

class ConfidenceBasedDetector:
    def __init__(self, model, confidence_threshold=0.25):
        self.model = model
        self.confidence_threshold = confidence_threshold
    
    def predict_with_progressive_confidence(self, image, progress_ratio):
        """
        Erweiterte Progressive Confidence-Berechnung mit Gaussian Uncertainty
        """
        results = self.model(image)
        enhanced_detections = []
        
        # Progressive confidence adjustment
        alpha = 5.0
        delta = 2 / (1 + math.exp(-alpha * progress_ratio)) - 1
        
        for detection in results[0].boxes:
            if detection.cls.item() == 6:  # Skip specimen_bag
                continue
            
            class_id = int(detection.cls.item())
            detector_confidence = detection.conf.item()
            box = detection.xyxy[0].cpu().numpy()
            
            # Gaussian-basierte Box Uncertainty
            box_uncertainty = self.model.predict_with_gaussian_uncertainty(image, box)
            box_confidence = 1 - box_uncertainty
            
            # Klassenspezifischer Threshold
            class_threshold = self.get_class_specific_threshold(
                class_id, 
                progress_ratio
            )
            
            # Combined confidence mit Klassengewichtung
            tool_weight = TOOL_MAPPING[class_id]['weight']
            combined_confidence = (
                (1 - delta) * detector_confidence + 
                delta * (detector_confidence * box_confidence)
            ) * tool_weight
            
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
    
    def get_class_specific_threshold(self, class_id, progress_ratio):
        """Berechnet klassenspezifische Schwellenwerte"""
        base_threshold = TOOL_MAPPING[class_id]['base_threshold']
        weight = TOOL_MAPPING[class_id]['weight']
        
        # Niedrigerer Threshold für schwache Klassen am Anfang
        if weight > 0.5:  # Schwache Klassen
            threshold = base_threshold * (0.5 + 0.5 * progress_ratio)
        else:
            threshold = base_threshold
            
        return threshold

class ConfMixDetector:
    def __init__(self, confidence_detector):
        self.confidence_detector = confidence_detector
    
    def process_frame(self, target_image, progress_ratio):
        """
        Erweiterte ConfMix Region Selection Strategy
        """
        # Ensure correct image size
        if target_image.size != IMAGE_SIZE:
            target_image = target_image.resize(IMAGE_SIZE)
            
        regions = self._split_image_into_regions(target_image)
        detections = self.confidence_detector.predict_with_progressive_confidence(
            target_image, 
            progress_ratio
        )
        
        region_confidences = self._calculate_region_confidences(detections, regions)
        
        # Adaptive Region Selection basierend auf Training Progress
        if progress_ratio < 0.3:
            # Frühe Phase: Fokus auf hohe Confidence
            best_regions = self._select_high_confidence_regions(region_confidences)
        else:
            # Späte Phase: Berücksichtige auch Anzahl der Detektionen
            best_regions = self._select_balanced_regions(region_confidences)
            
        return {
            'regions': regions,
            'detections': detections,
            'region_confidences': region_confidences,
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
    
    def _calculate_region_confidences(self, detections, regions):
        """Berechnet gewichtete Region-Confidences"""
        region_confidences = []
        
        for region in regions:
            region_dets = []
            weighted_conf = 0.0
            class_counts = {}
            
            for det in detections:
                box = det['box']
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                
                if (region[0] <= center_x <= region[2] and 
                    region[1] <= center_y <= region[3]):
                    region_dets.append(det)
                    class_id = det['class']
                    
                    # Gewichtete Confidence
                    weight = TOOL_MAPPING[class_id]['weight']
                    weighted_conf += det['combined_confidence'] * weight
                    
                    # Zähle Klassen
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
            
            mean_conf = weighted_conf / len(region_dets) if region_dets else 0
            
            region_confidences.append({
                'detections': region_dets,
                'mean_confidence': mean_conf,
                'class_counts': class_counts,
                'total_detections': len(region_dets)
            })
            
        return region_confidences
    
    def _select_high_confidence_regions(self, region_confidences, top_k=2):
        """Wählt Regionen mit höchster Confidence"""
        sorted_regions = sorted(
            range(len(region_confidences)),
            key=lambda i: region_confidences[i]['mean_confidence'],
            reverse=True
        )
        return sorted_regions[:top_k]
    
    def _select_balanced_regions(self, region_confidences, top_k=2):
        """Wählt Regionen basierend auf Confidence und Klassenverteilung"""
        region_scores = []
        
        for conf in region_confidences:
            # Basis-Score ist die mittlere Confidence
            score = conf['mean_confidence']
            
            # Bonus für unterrepräsentierte Klassen
            for class_id, count in conf['class_counts'].items():
                if TOOL_MAPPING[class_id]['weight'] > 0.5:  # Schwache Klassen
                    score *= (1 + 0.2 * count)
            
            region_scores.append(score)
            
        # Wähle Top-K Regionen
        sorted_regions = sorted(
            range(len(region_scores)),
            key=lambda i: region_scores[i],
            reverse=True
        )
        return sorted_regions[:top_k]
    def create_mixed_sample(self, target_image, source_image, target_results):
        """Erstellt gemischtes Sample mit Multi-Region Support"""
        if target_image.size != IMAGE_SIZE:
            target_image = target_image.resize(IMAGE_SIZE)
        if source_image.size != IMAGE_SIZE:
            source_image = source_image.resize(IMAGE_SIZE)
            
        # Erstelle Basis-Maske
        mask = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]))
        
        # Fülle ausgewählte Regionen
        for region_idx in target_results['best_regions']:
            x1, y1, x2, y2 = target_results['regions'][region_idx]
            mask[y1:y2, x1:x2] = 1
        
        # Mix images
        target_array = np.array(target_image)
        source_array = np.array(source_image)
        mixed_array = source_array.copy()
        
        # Ersetze ausgewählte Regionen
        for region_idx in target_results['best_regions']:
            x1, y1, x2, y2 = target_results['regions'][region_idx]
            mixed_array[y1:y2, x1:x2] = target_array[y1:y2, x1:x2]
        
        # Berechne durchschnittliche Confidence für ausgewählte Regionen
        mean_confidence = np.mean([
            target_results['region_confidences'][idx]['mean_confidence']
            for idx in target_results['best_regions']
        ]) if target_results['best_regions'] else 0.0

        # Sammle alle relevanten Detektionen aus den ausgewählten Regionen
        selected_detections = []
        for region_idx in target_results['best_regions']:
            selected_detections.extend(
                target_results['region_confidences'][region_idx]['detections']
            )
        
        return {
            'mixed_image': Image.fromarray(mixed_array.astype('uint8')),
            'mixing_mask': mask,
            'selected_regions': [
                target_results['regions'][idx] 
                for idx in target_results['best_regions']
            ],
            'confidence': mean_confidence,
            'target_detections': selected_detections
        }

    def get_valid_detections(self, detections, region):
        """Filtert valide Detektionen für eine Region"""
        x1, y1, x2, y2 = region
        valid_dets = []
        
        for det in detections:
            box = det['box']
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            
            if (x1 <= center_x <= x2 and y1 <= center_y <= y2):
                valid_dets.append(det)
                
        return valid_dets

    def calculate_class_distribution(self, detections):
        """Berechnet Klassenverteilung in Detektionen"""
        class_counts = {}
        total_dets = len(detections)
        
        for det in detections:
            class_id = det['class']
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            
        # Berechne Verteilung in Prozent
        class_distribution = {
            class_id: (count / total_dets if total_dets > 0 else 0)
            for class_id, count in class_counts.items()
        }
        
        return class_distribution

    def calculate_weak_class_ratio(self, detections):
        """Berechnet Verhältnis von schwachen zu starken Klassen"""
        weak_count = 0
        strong_count = 0
        
        for det in detections:
            if TOOL_MAPPING[det['class']]['weight'] > 0.5:
                weak_count += 1
            else:
                strong_count += 1
                
        total = weak_count + strong_count
        return weak_count / total if total > 0 else 0

