import os
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
from pathlib import Path
import random

# Erweiterte Constants
YOLO_MODEL_PATH = "/data/Bartscht/YOLO/best_v35.pt"
HEICHOLE_DATASET_PATH = "/data/Bartscht/HeiChole/domain_adaptation/test"
CHOLECT50_PATH = "/data/Bartscht/YOLO"
MIXED_SAMPLES_PATH = "/data/Bartscht/mixed_samples"  
CONFIDENCE_THRESHOLD = 0.25

# Instrument mappings from your existing code
TOOL_MAPPING = {
    0: 'grasper', 1: 'bipolar', 2: 'hook', 
    3: 'scissors', 4: 'clipper', 5: 'irrigator'
}

class ConfidenceBasedDetector:
    def __init__(self, model_path, confidence_threshold=0.25):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
    def predict_with_gaussian_confidence(self, image):
        results = self.model(image)
        enhanced_detections = []
        
        for detection in results[0].boxes:
            if detection.cls.item() == 6:  # Skip specimen_bag
                continue
                
            detector_confidence = detection.conf.item()
            box = detection.xyxy[0].cpu().numpy()
            
            box_variance = self._calculate_box_variance(box)
            box_confidence = 1 - np.mean(box_variance)
            combined_confidence = detector_confidence * box_confidence
            
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
        
    def process_frame(self, target_image):
        # 1. Get regions
        regions = self._split_image_into_regions(target_image)
        
        # 2. Get detections with confidence
        detections = self.confidence_detector.predict_with_gaussian_confidence(target_image)
        
        # 3. Calculate confidence per region
        region_confidences = self._calculate_region_confidences(detections, regions)
        
        # 4. Select best region
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

def main():
    """Hauptfunktion zur Erstellung der gemischten Trainingsdaten"""
    # Initialisierung
    confidence_detector = ConfidenceBasedDetector(YOLO_MODEL_PATH)
    confmix_detector = ConfMixDetector(confidence_detector)
    trainer = ConfMixTrainer(confmix_detector, CHOLECT50_PATH)
    
    # Output-Verzeichnisse erstellen
    os.makedirs(MIXED_SAMPLES_PATH, exist_ok=True)
    os.makedirs(MIXED_SAMPLES_PATH + "/images", exist_ok=True)
    os.makedirs(MIXED_SAMPLES_PATH + "/labels", exist_ok=True)
    
    # Counter für erfolgreiche Mixings
    successful_mixes = 0
    
    # HeiChole Dataset verarbeiten
    dataset_path = Path(HEICHOLE_DATASET_PATH)
    for video_folder in (dataset_path / "Videos").iterdir():
        if not video_folder.is_dir():
            continue
            
        print(f"\nProcessing HeiChole video: {video_folder.name}")
        
        for frame_file in video_folder.glob("*.png"):
            # HeiChole Bild laden
            target_image = Image.open(frame_file)
            
            # ConfMix Regionen und Konfidenz berechnen
            target_results = confmix_detector.process_frame(target_image)
            
            # Gemischtes Sample erstellen wenn Konfidenz gut genug
            mixed_sample = trainer.create_mixed_sample(target_image, target_results)
            
            if mixed_sample is not None:
                # Speichere gemischtes Bild
                mixed_image_path = os.path.join(MIXED_SAMPLES_PATH, "images", f"mixed_{successful_mixes:06d}.png")
                mixed_sample['mixed_image'].save(mixed_image_path)
                
                # Speichere Meta-Informationen
                meta_path = os.path.join(MIXED_SAMPLES_PATH, "labels", f"mixed_{successful_mixes:06d}.txt")
                with open(meta_path, 'w') as f:
                    f.write(f"Source: {mixed_sample['source_data']['path'].name}\n")
                    f.write(f"Target region: {mixed_sample['target_region']}\n")
                    f.write(f"Confidence: {mixed_sample['confidence']}\n")
                    f.write("Source labels:\n")
                    f.writelines(mixed_sample['source_data']['labels'])
                    f.write("\nTarget detections:\n")
                    for det in mixed_sample['target_detections']:
                        f.write(f"{det['class']} {det['combined_confidence']} {det['box']}\n")
                
                successful_mixes += 1
                if successful_mixes % 100 == 0:
                    print(f"Created {successful_mixes} mixed samples")

if __name__ == "__main__":
    main()