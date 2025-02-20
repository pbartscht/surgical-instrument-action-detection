import os
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
import random
import math
from collections import defaultdict

# Konstanten
TOOL_MAPPING = {
    0: 'grasper', 1: 'bipolar', 2: 'hook', 
    3: 'scissors', 4: 'clipper', 5: 'irrigator', 6: 'specimenBag'
}

# Adaptive Schwellenwerte für problematische Klassen
CLASS_CONFIDENCE_THRESHOLDS = {
    3: 0.05,  # scissors
    4: 0.05,  # clipper
    5: 0.05,  # irrigator
    'default': 0.09
}

class EnhancedConfidenceDetector:
    def __init__(self, model_path, use_tta=True, multi_scale=True):
        self.model = YOLO(model_path)
        self.use_tta = use_tta
        self.multi_scale = multi_scale
        self.class_thresholds = CLASS_CONFIDENCE_THRESHOLDS
        
    def predict_with_uncertainty(self, image, progress_ratio):
        """Erweiterte Vorhersage mit TTA und Multi-Scale Detection"""
        all_detections = []
        
        # Multi-Scale Detection
        if self.multi_scale:
            scales = [0.8, 1.0, 1.2]
        else:
            scales = [1.0]
            
        for scale in scales:
            scaled_size = (int(image.width * scale), int(image.height * scale))
            scaled_img = image.resize(scaled_size, Image.Resampling.LANCZOS)
            
            # Test-Time Augmentation
            if self.use_tta:
                aug_images = self._generate_augmentations(scaled_img)
            else:
                aug_images = [scaled_img]
            
            # Process each augmented version
            for aug_img in aug_images:
                detections = self._process_single_image(aug_img, scale, progress_ratio)
                all_detections.extend(detections)
        
        # Merge overlapping detections
        merged_detections = self._merge_overlapping_detections(all_detections)
        return merged_detections
    
    def _process_single_image(self, image, scale, progress_ratio):
        """Verarbeitet ein einzelnes Bild"""
        results = self.model(image)
        detections = []
        
        for detection in results[0].boxes:
            if detection.cls.item() == 6:  # Skip specimenBag
                continue
                
            class_id = int(detection.cls.item())
            confidence = float(detection.conf.item())
            box = detection.xyxy[0].cpu().numpy()
            
            # Skaliere Box zurück wenn nötig
            if scale != 1.0:
                box = box / scale
            
            # Adaptive Confidence Threshold
            threshold = self.class_thresholds.get(class_id, 
                                                self.class_thresholds['default'])
            
            # Progressive Confidence Adjustment
            box_uncertainty = self._calculate_box_uncertainty(box)
            adjusted_conf = self._adjust_confidence(confidence, box_uncertainty, 
                                                 progress_ratio)
            
            if adjusted_conf >= threshold:
                detections.append({
                    'box': box,
                    'class': class_id,
                    'confidence': confidence,
                    'uncertainty': box_uncertainty,
                    'adjusted_confidence': adjusted_conf,
                    'instrument_name': TOOL_MAPPING[class_id]
                })
        
        return detections
    
    def _generate_augmentations(self, image):
        """Generiert TTA Varianten"""
        augmentations = [image]  # Original
        
        # Horizontal flip
        augmentations.append(image.transpose(Image.FLIP_LEFT_RIGHT))
        
        # Slight rotations
        for angle in [-5, 5]:
            augmentations.append(image.rotate(angle, expand=False))
        
        # Brightness variation
        enhancer = ImageEnhance.Brightness(image)
        augmentations.append(enhancer.enhance(1.1))
        
        return augmentations
    
    def _calculate_box_uncertainty(self, box):
        """Berechnet Uncertainty basierend auf Box-Eigenschaften"""
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        # Größen-basierte Uncertainty
        size_uncertainty = 1 / (1 + width * height)
        
        # Aspekt-Ratio Uncertainty
        aspect_ratio = max(width, height) / min(width, height)
        ratio_uncertainty = 1 / (1 + math.exp(-(aspect_ratio - 2)))
        
        return 0.7 * size_uncertainty + 0.3 * ratio_uncertainty
    
    def _adjust_confidence(self, confidence, uncertainty, progress_ratio):
        """Passt Confidence basierend auf Uncertainty und Training Progress an"""
        # Progressive adjustment factor
        delta = 2 / (1 + math.exp(-5 * progress_ratio)) - 1
        
        # Combine confidence and uncertainty
        adjusted = (1 - delta) * confidence + delta * (1 - uncertainty)
        
        return adjusted
    
    def _merge_overlapping_detections(self, detections, iou_threshold=0.5):
        """Merged überlappende Detektionen mit verbesserter Strategie"""
        if not detections:
            return []
        
        # Sort by adjusted confidence
        detections = sorted(detections, 
                          key=lambda x: x['adjusted_confidence'], 
                          reverse=True)
        
        merged = []
        while detections:
            best_det = detections.pop(0)
            overlapping_same_class = []
            i = 0
            
            # Find overlapping boxes of same class
            while i < len(detections):
                if (detections[i]['class'] == best_det['class'] and 
                    self._compute_iou(best_det['box'], detections[i]['box']) > iou_threshold):
                    overlapping_same_class.append(detections.pop(i))
                else:
                    i += 1
            
            if overlapping_same_class:
                # Weighted average of boxes and confidences
                all_boxes = np.vstack([best_det['box'][None, :]] + 
                                    [d['box'][None, :] for d in overlapping_same_class])
                all_confs = np.array([best_det['adjusted_confidence']] + 
                                   [d['adjusted_confidence'] for d in overlapping_same_class])
                
                weights = all_confs / all_confs.sum()
                merged_box = (all_boxes * weights[:, None]).sum(axis=0)
                merged_conf = all_confs.mean()
                merged_uncertainty = np.mean([best_det['uncertainty']] + 
                                          [d['uncertainty'] for d in overlapping_same_class])
                
                merged.append({
                    'box': merged_box,
                    'class': best_det['class'],
                    'confidence': merged_conf,
                    'uncertainty': merged_uncertainty,
                    'instrument_name': best_det['instrument_name']
                })
            else:
                merged.append(best_det)
        
        return merged
    
    def _compute_iou(self, box1, box2):
        """Berechnet IoU zwischen zwei Boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        return intersection / union if union > 0 else 0

class RegionAnalyzer:
    def __init__(self, image_size):
        self.image_size = image_size
        self.regions = self._create_regions()
        
    def _create_regions(self):
        """Erstellt vier gleichgroße Regionen"""
        width, height = self.image_size
        return [
            (0, 0, width//2, height//2),
            (width//2, 0, width, height//2),
            (0, height//2, width//2, height),
            (width//2, height//2, width, height)
        ]
    
    def analyze_regions(self, detections):
        """Analysiert alle Regionen und ihre Detektionen"""
        region_analyses = []
        
        for region in self.regions:
            analysis = self._analyze_single_region(region, detections)
            region_analyses.append(analysis)
        
        return region_analyses
    
    def _analyze_single_region(self, region, detections):
        """Analysiert eine einzelne Region"""
        region_dets = []
        class_counts = defaultdict(int)
        total_confidence = 0
        total_uncertainty = 0
        
        for det in detections:
            box = det['box']
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            
            if (region[0] <= center_x <= region[2] and 
                region[1] <= center_y <= region[3]):
                region_dets.append(det)
                class_counts[det['class']] += 1
                total_confidence += det['adjusted_confidence']
                total_uncertainty += det['uncertainty']
        
        if region_dets:
            mean_conf = total_confidence / len(region_dets)
            mean_uncertainty = total_uncertainty / len(region_dets)
            quality_score = self._calculate_quality_score(
                mean_conf, mean_uncertainty, len(region_dets), class_counts
            )
        else:
            mean_conf = 0
            mean_uncertainty = 1
            quality_score = 0
        
        return {
            'region': region,
            'detections': region_dets,
            'class_distribution': dict(class_counts),
            'mean_confidence': mean_conf,
            'mean_uncertainty': mean_uncertainty,
            'quality_score': quality_score
        }
    
    def _calculate_quality_score(self, confidence, uncertainty, num_dets, class_counts):
        """Berechnet Quality Score für die Region"""
        confidence_score = confidence * (1 - uncertainty)
        density_score = min(num_dets / 4, 1.0)  # Max 4 instruments per region
        diversity_score = len(class_counts) / len(TOOL_MAPPING)
        
        # Weighted combination
        return (0.4 * confidence_score + 
                0.3 * density_score + 
                0.3 * diversity_score)
    
    def get_best_region(self, analyses):
        """Wählt die beste Region basierend auf Quality Score"""
        return max(analyses, key=lambda x: x['quality_score'])