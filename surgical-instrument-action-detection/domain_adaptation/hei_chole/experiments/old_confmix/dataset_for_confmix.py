import os
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
from pathlib import Path
import random
import math

# Erweiterte Constants
YOLO_MODEL_PATH = "/data/Bartscht/YOLO/best_v35.pt"
HEICHOLE_DATASET_PATH = "/data/Bartscht/HeiChole/domain_adaptation/train"
CHOLECT50_PATH = "/data/Bartscht/YOLO"
MIXED_SAMPLES_PATH = "/data/Bartscht/mixed_samples_epoch0"  
CONFIDENCE_THRESHOLD = 0.1

# Instrument mappings from your existing code
TOOL_MAPPING = {
    0: 'grasper', 1: 'bipolar', 2: 'hook', 
    3: 'scissors', 4: 'clipper', 5: 'irrigator'
}

class ConfidenceBasedDetector:
    def __init__(self, model_path, confidence_threshold=0.25):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
    def predict_with_progressive_confidence(self, image, progress_ratio):
        """
        Macht Vorhersagen mit progressiver Confidence
        progress_ratio: Float zwischen 0-1, der den Fortschritt angibt
        """
        results = self.model(image)
        enhanced_detections = []
        
        # Berechne delta für progressive confidence
        delta = 2 / (1 + math.exp(-5 * progress_ratio)) - 1
        
        for detection in results[0].boxes:
            if detection.cls.item() == 6:  # Skip specimen_bag
                continue
                
            # Detector confidence (Cdet)
            detector_confidence = detection.conf.item()
            box = detection.xyxy[0].cpu().numpy()
            
            # Box confidence (Ccomb)
            box_variance = self._calculate_box_variance(box)
            box_confidence = 1 - np.mean(box_variance)
            
            # Progressive combination
            combined_confidence = (1 - delta) * detector_confidence + delta * (detector_confidence * box_confidence)
            
            if combined_confidence >= self.confidence_threshold:
                enhanced_detections.append({
                    'box': box,
                    'class': int(detection.cls.item()),
                    'detector_confidence': detector_confidence,
                    'box_confidence': box_confidence,
                    'combined_confidence': combined_confidence,
                    'instrument_name': TOOL_MAPPING[int(detection.cls.item())]
                })
        
        return enhanced_detections
    
    def _calculate_box_variance(self, box):
        # Diese Methode bleibt unverändert
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        size_variance = np.array([
            1 / (1 + width * height),
            1 / (1 + min(width, height))
        ])
        
        return size_variance

class ConfMixDetector:
    def __init__(self, confidence_detector):
        self.confidence_detector = confidence_detector
        
    def process_frame(self, target_image, progress_ratio):
        # 1. Get regions
        regions = self._split_image_into_regions(target_image)
        
        # 2. Get detections with progressive confidence
        detections = self.confidence_detector.predict_with_progressive_confidence(
            target_image, 
            progress_ratio
        )
        
        # Rest bleibt gleich
        region_confidences = self._calculate_region_confidences(detections, regions)
        best_region_idx = np.argmax([conf['mean_confidence'] for conf in region_confidences])
        
        return {
            'regions': regions,
            'detections': detections,
            'region_confidences': region_confidences,
            'best_region_idx': best_region_idx
        }
    
    
    def _split_image_into_regions(self, image):
        width, height = image.size
        regions = [
            (0, 0, width//2, height//2),
            (width//2, 0, width, height//2),
            (0, height//2, width//2, height),
            (width//2, height//2, width, height)
        ]
        return regions
    
    def _calculate_region_confidences(self, detections, regions):
        region_confidences = []
        
        for region in regions:
            region_dets = []
            region_conf = 0.0
            
            for det in detections:
                box = det['box']
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                
                if (region[0] <= center_x <= region[2] and 
                    region[1] <= center_y <= region[3]):
                    region_dets.append(det)
                    region_conf += det['combined_confidence']
            
            mean_conf = region_conf / len(region_dets) if region_dets else 0
            region_confidences.append({
                'detections': region_dets,
                'mean_confidence': mean_conf
            })
            
        return region_confidences

def process_heichole_dataset():
    """Main function to process HeiChole dataset"""
    # Initialize detectors
    confidence_detector = ConfidenceBasedDetector(YOLO_MODEL_PATH)
    confmix_detector = ConfMixDetector(confidence_detector)
    
    # Process each video in the dataset
    dataset_path = Path(HEICHOLE_DATASET_PATH)
    for video_folder in (dataset_path / "Videos").iterdir():
        if not video_folder.is_dir():
            continue
            
        print(f"\nProcessing video: {video_folder.name}")
        
        # Process each frame in the video
        for frame_file in video_folder.glob("*.png"):
            # Load image
            image = Image.open(frame_file)
            
            # Process frame with ConfMix
            results = confmix_detector.process_frame(image)
            
            # Print detections for testing
            print(f"\nFrame {frame_file.stem}:")
            print(f"Found {len(results['detections'])} detections")
            print(f"Best region: {results['best_region_idx']}")
            print(f"Region confidences: {[conf['mean_confidence'] for conf in results['region_confidences']]}")

class ConfMixTrainer:
    def __init__(self, confmix_detector, source_data_path):
        self.confmix_detector = confmix_detector
        self.source_images_path = Path(source_data_path) / "images" / "train"
        self.source_labels_path = Path(source_data_path) / "labels" / "train"
        
        # Validiere Pfade
        if not self.source_images_path.exists():
            raise FileNotFoundError(f"Source images path not found: {self.source_images_path}")
            
        # Liste alle verfügbaren CholecT50-Bilder
        self.source_images = list(self.source_images_path.glob("*.png"))
        print(f"Found {len(self.source_images)} CholecT50 source images")
    
    def _get_random_source_image(self):
        """Lädt zufälliges CholecT50-Bild mit Labels"""
        source_image_path = random.choice(self.source_images)
        source_image = Image.open(source_image_path)
        
        # Lade zugehöriges YOLO-Label
        label_path = self.source_labels_path / (source_image_path.stem + ".txt")
        source_labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                source_labels = f.readlines()
        
        return {
            'image': source_image,
            'path': source_image_path,
            'labels': source_labels
        }

    def create_mixed_sample(self, target_image, target_results):
        """Erstellt gemischtes Bild nach ConfMix-Methode"""
        source_data = self._get_random_source_image()
        source_image = source_data['image']
        
        # Resize falls nötig
        if source_image.size != target_image.size:
            source_image = source_image.resize(target_image.size)
        
        # Beste Region aus HeiChole
        best_region_idx = target_results['best_region_idx']
        best_region = target_results['regions'][best_region_idx]
        confidence = target_results['region_confidences'][best_region_idx]['mean_confidence']
        
        # Nur mixen wenn Konfidenz gut genug
        if confidence > CONFIDENCE_THRESHOLD:
            # Mixing-Maske erstellen
            mask = np.zeros((target_image.size[1], target_image.size[0]))
            x1, y1, x2, y2 = best_region
            mask[y1:y2, x1:x2] = 1
            
            # Bilder mixen
            target_array = np.array(target_image)
            source_array = np.array(source_image)
            mixed_array = (1 - mask[..., None]) * source_array + mask[..., None] * target_array
            mixed_image = Image.fromarray(mixed_array.astype('uint8'))
            
            return {
                'mixed_image': mixed_image,
                'source_data': source_data,
                'target_region': best_region,
                'confidence': confidence,
                'target_detections': target_results['detections']
            }
        return None

def convert_to_yolo_format(box, image_width, image_height):
    """Konvertiert Box-Koordinaten ins YOLO-Format"""
    x1, y1, x2, y2 = box
    
    # Berechne zentrale Koordinaten und Dimensionen
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    # Normalisiere auf Bildgröße
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height
    
    return x_center, y_center, width, height
def save_combined_labels(mixed_sample, label_path, image_width, image_height):
    """
    Speichert kombinierte Labels aus Source und Target
    basierend auf der gemixten Region
    """
    target_region = mixed_sample['target_region']
    x1, y1, x2, y2 = target_region
    
    combined_labels = []
    
    # 1. Target-Pseudo-Labels (nur für die ausgewählte Region)
    for det in mixed_sample['target_detections']:
        box = det['box']
        box_center_x = (box[0] + box[2]) / 2
        box_center_y = (box[1] + box[3]) / 2
        
        # Prüfe ob die Detection in der Target-Region liegt
        if (x1 <= box_center_x <= x2 and y1 <= box_center_y <= y2):
            x_center, y_center, width, height = convert_to_yolo_format(
                box, image_width, image_height
            )
            combined_labels.append({
                'class': det['class'],
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
    
    # 2. Source-Labels (für den Rest des Bildes)
    for label in mixed_sample['source_data']['labels']:
        # Parse YOLO-Format Label
        parts = label.strip().split()
        if len(parts) == 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Konvertiere zu absoluten Koordinaten
            abs_x = x_center * image_width
            abs_y = y_center * image_height
            
            # Prüfe ob das Label außerhalb der Target-Region liegt
            if not (x1 <= abs_x <= x2 and y1 <= abs_y <= y2):
                combined_labels.append({
                    'class': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
    
    # Speichere alle Labels
    with open(label_path, 'w') as f:
        for label in combined_labels:
            f.write(f"{label['class']} {label['x_center']:.6f} "
                   f"{label['y_center']:.6f} {label['width']:.6f} "
                   f"{label['height']:.6f}\n")

def main():
    """Hauptfunktion zur Erstellung der gemischten Trainingsdaten"""
    confidence_detector = ConfidenceBasedDetector(YOLO_MODEL_PATH)
    confmix_detector = ConfMixDetector(confidence_detector)
    trainer = ConfMixTrainer(confmix_detector, CHOLECT50_PATH)
    
    # Erweiterte Verzeichnisstruktur
    os.makedirs(MIXED_SAMPLES_PATH, exist_ok=True)
    os.makedirs(os.path.join(MIXED_SAMPLES_PATH, "images"), exist_ok=True)
    os.makedirs(os.path.join(MIXED_SAMPLES_PATH, "labels"), exist_ok=True)
    os.makedirs(os.path.join(MIXED_SAMPLES_PATH, "labels", "meta"), exist_ok=True)
    
    successful_mixes = 0
    total_frames = sum(1 for _ in Path(HEICHOLE_DATASET_PATH).rglob("*.png"))
    current_frame = 0
    
    dataset_path = Path(HEICHOLE_DATASET_PATH)
    for video_folder in (dataset_path / "Videos").iterdir():
        if not video_folder.is_dir():
            continue
            
        print(f"\nProcessing HeiChole video: {video_folder.name}")
        
        for frame_file in video_folder.glob("*.png"):
            progress_ratio = current_frame / total_frames
            current_frame += 1
            
            target_image = Image.open(frame_file)
            target_results = confmix_detector.process_frame(target_image, progress_ratio)
            mixed_sample = trainer.create_mixed_sample(target_image, target_results)
            
            if mixed_sample is not None:
                # Basis-Dateinamen generieren
                base_filename = f"mixed_{successful_mixes:06d}"
                
                # Pfade für Bild, Label und Meta
                image_path = os.path.join(MIXED_SAMPLES_PATH, "images", f"{base_filename}.png")
                label_path = os.path.join(MIXED_SAMPLES_PATH, "labels", f"{base_filename}.txt")
                meta_path = os.path.join(MIXED_SAMPLES_PATH, "labels", "meta", f"{base_filename}_meta.txt")
                
                # Speichere Bild
                mixed_sample['mixed_image'].save(image_path)
                
                # Speichere kombinierte Labels
                image_width, image_height = mixed_sample['mixed_image'].size
                save_combined_labels(mixed_sample, label_path, image_width, image_height)
                
                # Speichere Meta-Informationen
                with open(meta_path, 'w') as f:
                    f.write(f"Progress Ratio: {progress_ratio:.3f}\n")
                    f.write(f"Source: {mixed_sample['source_data']['path'].name}\n")
                    f.write(f"Target region: {mixed_sample['target_region']}\n")
                    f.write(f"Confidence: {mixed_sample['confidence']}\n")
                    f.write("Original detections:\n")
                    for det in mixed_sample['target_detections']:
                        f.write(f"{det['class']} {det['combined_confidence']} {det['box']}\n")
                
                successful_mixes += 1
                if successful_mixes % 100 == 0:
                    print(f"Created {successful_mixes} mixed samples (Progress: {progress_ratio:.2%})")

if __name__ == "__main__":
    main()