import os
import yaml
from pathlib import Path
from tqdm import tqdm
import json
from collections import Counter, defaultdict
import statistics
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont, ImageFilter 
from pathlib import Path
import random
import math
from core_components import EnhancedConfidenceDetector, RegionAnalyzer, TOOL_MAPPING


# Pfade und Konstanten
YOLO_MODEL_PATH = "/data/Bartscht/YOLO/best_v35.pt"
HEICHOLE_DATASET_PATH = "/data/Bartscht/HeiChole/domain_adaptation/train"
CHOLECT50_PATH = "/data/Bartscht/YOLO"
BALANCED_MIXED_SAMPLES_PATH = "/data/Bartscht/balanced_mixed_samples_epoch_last"

class DatasetCreator:
    def __init__(self, model_path, source_path, target_path, output_path):
        self.detector = EnhancedConfidenceDetector(
            model_path,
            use_tta=True,
            multi_scale=True
        )
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.output_path = Path(output_path)
        
        # Initialisiere Tracking-Statistiken
        self.class_stats = defaultdict(lambda: {
            'count': 0,
            'confidence_sum': 0,
            'uncertainty_sum': 0,
            'false_negatives': 0,
            'quality_scores': []
        })
        
        self._setup_directories()
        self._load_source_images()
    
    def _setup_directories(self):
        """Erstellt notwendige Verzeichnisstruktur"""
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels', 'labels/meta']:
                path = self.output_path / subdir / split
                path.mkdir(parents=True, exist_ok=True)
    
    def _load_source_images(self):
        """Lädt verfügbare Source-Bilder"""
        self.source_images = list((self.source_path / "images" / "train").glob("*.png"))
        print(f"Loaded {len(self.source_images)} source images")
    
    def create_dataset(self, max_samples_per_class=3000):
        """Erstellt verbessertes Mixed-Sample Dataset"""
        target_videos = list((self.target_path / "Videos").glob("*"))
        total_frames = sum(len(list(video.glob("*.png"))) for video in target_videos)
        processed_count = 0
        
        for video_path in target_videos:
            if not video_path.is_dir():
                continue
            
            print(f"\nProcessing video: {video_path.name}")
            for frame_path in tqdm(list(video_path.glob("*.png"))):
                progress_ratio = processed_count / total_frames
                
                mixed_sample = self._process_frame(frame_path, progress_ratio)
                if mixed_sample and self._validate_sample(mixed_sample):
                    self._save_mixed_sample(mixed_sample, processed_count)
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        self._log_statistics()
        
        self._create_final_dataset_config()
    
    def _process_frame(self, frame_path, progress_ratio):
        """Verarbeitet einzelnes Frame mit verbesserter Analyse"""
        target_image = Image.open(frame_path)
        
        # Get detections with uncertainty estimation
        detections = self.detector.predict_with_uncertainty(target_image, progress_ratio)
        if not detections:
            return None
            
        # Analyze regions
        region_analyzer = RegionAnalyzer(target_image.size)
        region_analyses = region_analyzer.analyze_regions(detections)
        best_region = region_analyzer.get_best_region(region_analyses)
        
        if best_region['quality_score'] < 0.3:  # Minimum quality threshold
            return None
        
        # Get source image and create mix
        source_data = self._get_random_source_image()
        mixed_sample = self._create_mixed_sample(
            target_image, 
            source_data, 
            best_region,
            detections
        )
        
        return mixed_sample
    
    def _get_random_source_image(self):
        """Holt zufälliges Source-Bild mit Labels"""
        source_image_path = random.choice(self.source_images)
        source_image = Image.open(source_image_path)
        
        label_path = self.source_path / "labels" / "train" / f"{source_image_path.stem}.txt"
        source_labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                source_labels = f.readlines()
        
        return {
            'image': source_image,
            'path': source_image_path,
            'labels': source_labels
        }
    
    def _create_mixed_sample(self, target_image, source_data, region_info, detections):
        """Erstellt gemischtes Sample mit verbessertem Blending"""
        source_image = source_data['image']
        if source_image.size != target_image.size:
            source_image = source_image.resize(target_image.size)
        
        # Create soft mask for better blending
        mask = Image.new('L', target_image.size, 0)
        draw = ImageDraw.Draw(mask)
        x1, y1, x2, y2 = region_info['region']
        draw.rectangle([x1, y1, x2, y2], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=10))
        
        # Mix images
        mask_array = np.array(mask) / 255.0
        target_array = np.array(target_image)
        source_array = np.array(source_image)
        mixed_array = (mask_array[..., None] * target_array + 
                      (1 - mask_array[..., None]) * source_array)
        
        return {
            'mixed_image': Image.fromarray(mixed_array.astype('uint8')),
            'source_data': source_data,
            'region_info': region_info,
            'detections': detections
        }
    
    def _validate_sample(self, mixed_sample):
        """Validiert Mixed Sample anhand mehrerer Kriterien"""
        region_info = mixed_sample['region_info']
        detections = mixed_sample['detections']
        
        # Check class distribution
        class_counts = defaultdict(int)
        for det in detections:
            if self._is_detection_in_region(det, region_info['region']):
                class_counts[det['class']] += 1
                
                # Check class limits
                if self.class_stats[det['class']]['count'] >= 3000:
                    return False
        
        # Validate detection quality
        if not self._validate_detections(detections, region_info['region']):
            return False
        
        return True
    
    def _validate_detections(self, detections, region):
        """Prüft Qualität der Detektionen"""
        region_dets = [d for d in detections 
                      if self._is_detection_in_region(d, region)]
        
        if not region_dets:
            return False
        
        # Check confidence and uncertainty
        mean_conf = np.mean([d['adjusted_confidence'] for d in region_dets])
        mean_uncertainty = np.mean([d['uncertainty'] for d in region_dets])
        
        return mean_conf > 0.3 and mean_uncertainty < 0.4
    
    def _is_detection_in_region(self, detection, region):
        """Prüft ob Detection in Region liegt"""
        box = detection['box']
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        x1, y1, x2, y2 = region
        return (x1 <= center_x <= x2 and y1 <= center_y <= y2)
    
    def _save_mixed_sample(self, mixed_sample, index):
        """Speichert Mixed Sample mit erweiterten Metadaten"""
        # Bestimme Split (80/10/10)
        r = random.random()
        split = "train" if r < 0.8 else "val" if r < 0.9 else "test"
        
        base_filename = f"mixed_{index:06d}"
        
        # Speichere Bild
        image_path = self.output_path / "images" / split / f"{base_filename}.png"
        mixed_sample['mixed_image'].save(image_path)
        
        # Speichere Labels
        self._save_labels(mixed_sample, split, base_filename)
        
        # Speichere erweiterte Metadaten
        self._save_metadata(mixed_sample, split, base_filename)
        
        # Update Statistiken
        self._update_statistics(mixed_sample)
    
    def _save_labels(self, mixed_sample, split, base_filename):
        """Speichert YOLO-Format Labels mit Qualitätskontrolle"""
        label_path = self.output_path / "labels" / split / f"{base_filename}.txt"
        region = mixed_sample['region_info']['region']
        image_size = mixed_sample['mixed_image'].size
        
        valid_labels = []
        
        # Process target region detections
        for det in mixed_sample['detections']:
            if self._is_detection_in_region(det, region):
                label = self._convert_to_yolo_format(det, image_size)
                if label:
                    valid_labels.append(label)
        
        # Process source labels outside target region
        for label_line in mixed_sample['source_data']['labels']:
            label = self._parse_source_label(label_line, region, image_size)
            if label:
                valid_labels.append(label)
        
        # Filter overlapping labels
        final_labels = self._filter_overlapping_labels(valid_labels)
        
        # Save labels
        with open(label_path, 'w') as f:
            for label in final_labels:
                f.write(f"{label['class']} {label['x']} {label['y']} "
                       f"{label['w']} {label['h']}\n")
    
    def _save_metadata(self, mixed_sample, split, base_filename):
        """Speichert erweiterte Metadaten für Analyse"""
        meta_path = self.output_path / "labels" / split / "meta" / f"{base_filename}.json"
        
        metadata = {
            'source_image': str(mixed_sample['source_data']['path'].name),
            'region': mixed_sample['region_info']['region'],
            'quality_score': mixed_sample['region_info']['quality_score'],
            'detections': []
        }
        
        for det in mixed_sample['detections']:
            metadata['detections'].append({
                'class': det['class'],
                'confidence': float(det['confidence']),
                'uncertainty': float(det['uncertainty']),
                'adjusted_confidence': float(det['adjusted_confidence']),
                'box': det['box'].tolist()
            })
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _update_statistics(self, mixed_sample):
        """Aktualisiert Tracking-Statistiken"""
        region = mixed_sample['region_info']['region']
        
        for det in mixed_sample['detections']:
            if self._is_detection_in_region(det, region):
                class_id = det['class']
                stats = self.class_stats[class_id]
                
                stats['count'] += 1
                stats['confidence_sum'] += det['confidence']
                stats['uncertainty_sum'] += det['uncertainty']
                stats['quality_scores'].append(det['adjusted_confidence'])
    
    def _log_statistics(self):
        """Loggt aktuelle Statistiken"""
        print("\nCurrent Dataset Statistics:")
        print("-" * 50)
        
        for class_id, stats in sorted(self.class_stats.items()):
            if stats['count'] > 0:
                mean_conf = stats['confidence_sum'] / stats['count']
                mean_uncertainty = stats['uncertainty_sum'] / stats['count']
                
                print(f"\nClass {TOOL_MAPPING[class_id]}:")
                print(f"  Count: {stats['count']}")
                print(f"  Mean Confidence: {mean_conf:.3f}")
                print(f"  Mean Uncertainty: {mean_uncertainty:.3f}")
    
    def _create_final_dataset_config(self):
        """Erstellt finale Dataset Konfiguration"""
        config = {
            'path': str(self.output_path),
            'train': str(self.output_path / 'images' / 'train'),
            'val': str(self.output_path / 'images' / 'val'),
            'test': str(self.output_path / 'images' / 'test'),
            'names': TOOL_MAPPING,
            'class_weights': self._calculate_class_weights()
        }
        
        yaml_path = self.output_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
    
    def _calculate_class_weights(self):
        """Berechnet optimierte Klassengewichte"""
        weights = {}
        total_samples = sum(stats['count'] for stats in self.class_stats.values())
        
        for class_id, stats in self.class_stats.items():
            if stats['count'] == 0:
                weights[class_id] = 1.0
                continue
            
            # Berücksichtige Klassenhäufigkeit und Detektionsqualität
            mean_quality = np.mean(stats['quality_scores'])
            count_factor = math.log(total_samples / stats['count'] + 1)
            
            weights[class_id] = count_factor * (1 + (1 - mean_quality))
        
        # Normalisiere Gewichte
        max_weight = max(weights.values())
        return {k: float(v/max_weight) for k, v in weights.items()}

def main():
    """Hauptfunktion zur Dataset-Erstellung"""
    try:
        creator = DatasetCreator(
            model_path=YOLO_MODEL_PATH,
            source_path=CHOLECT50_PATH,
            target_path=HEICHOLE_DATASET_PATH,
            output_path=BALANCED_MIXED_SAMPLES_PATH
        )
        
        print("Starting dataset creation...")
        creator.create_dataset(max_samples_per_class=3000)
        print("\nDataset creation completed successfully!")
        
    except Exception as e:
        print(f"Error during dataset creation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()