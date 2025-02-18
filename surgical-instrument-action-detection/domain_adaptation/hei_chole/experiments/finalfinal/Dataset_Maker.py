import os
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
from pathlib import Path
import random
import math
import yaml
from collections import Counter

# Konstanten und Pfade
YOLO_MODEL_PATH = "/data/Bartscht/YOLO/best_v35.pt"
HEICHOLE_DATASET_PATH = "/data/Bartscht/HeiChole/domain_adaptation/train"
CHOLECT50_PATH = "/data/Bartscht/YOLO"
# Ursprünglicher Ordner (wird nicht mehr genutzt) und neuer, balancierter Ordner
MIXED_SAMPLES_PATH = "/data/Bartscht/mixed_samples_epoch0"
BALANCED_MIXED_SAMPLES_PATH = "/data/Bartscht/balanced_mixed_samples_epoch0"

CONFIDENCE_THRESHOLD = 0.05
IOU_THRESHOLD = 0.5  # Schwellwert für Duplikat-Filterung

# Mapping der Instrumentklassen
TOOL_MAPPING = {
    0: 'grasper', 1: 'bipolar', 2: 'hook', 
    3: 'scissors', 4: 'clipper', 5: 'irrigator', 6: 'specimenBag'
}

# ---------------------------
# Helper-Funktionen für IoU & Duplikat-Filterung
# ---------------------------
def compute_iou(label1, label2):
    """
    Berechnet den Intersection-over-Union (IoU)-Wert zwischen zwei Boxen im YOLO-Format.
    label: dict mit 'x_center', 'y_center', 'width', 'height' (alle normiert).
    """
    # Umrechnung in Box-Koordinaten (links, oben, rechts, unten)
    x1_min = label1['x_center'] - label1['width'] / 2
    y1_min = label1['y_center'] - label1['height'] / 2
    x1_max = label1['x_center'] + label1['width'] / 2
    y1_max = label1['y_center'] + label1['height'] / 2

    x2_min = label2['x_center'] - label2['width'] / 2
    y2_min = label2['y_center'] - label2['height'] / 2
    x2_max = label2['x_center'] + label2['width'] / 2
    y2_max = label2['y_center'] + label2['height'] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    area1 = label1['width'] * label1['height']
    area2 = label2['width'] * label2['height']

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def filter_duplicate_labels(labels, iou_threshold=IOU_THRESHOLD):
    """
    Filtert überlappende Labels derselben Klasse heraus.
    Es wird jeweils das erste Vorkommen beibehalten; weitere Duplikate (IoU > iou_threshold) werden entfernt.
    """
    filtered = []
    for label in labels:
        duplicate = False
        for kept in filtered:
            if label['class'] == kept['class']:
                if compute_iou(label, kept) > iou_threshold:
                    duplicate = True
                    break
        if not duplicate:
            filtered.append(label)
    return filtered

# ---------------------------
# Detector- und Trainer-Klassen (basierend auf dem alten Code)
# ---------------------------
class ConfidenceBasedDetector:
    def __init__(self, model_path, confidence_threshold=0.25):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
    def predict_with_progressive_confidence(self, image, progress_ratio):
        """
        Macht Vorhersagen mit progressiver Confidence.
        progress_ratio: Float zwischen 0 und 1.
        """
        results = self.model(image)
        enhanced_detections = []
        
        # Berechne delta für progressive Confidence
        delta = 2 / (1 + math.exp(-5 * progress_ratio)) - 1
        
        for detection in results[0].boxes:
            if detection.cls.item() == 6:  # specimenBag überspringen
                continue
                
            # Detector confidence (Cdet)
            detector_confidence = detection.conf.item()
            box = detection.xyxy[0].cpu().numpy()
            
            # Box confidence (Ccomb)
            box_variance = self._calculate_box_variance(box)
            box_confidence = 1 - np.mean(box_variance)
            
            # Progressive Kombination
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
        # Aufteilen des Bildes in vier Regionen
        regions = self._split_image_into_regions(target_image)
        
        # Detektionen mit progressiver Confidence erhalten
        detections = self.confidence_detector.predict_with_progressive_confidence(target_image, progress_ratio)
        
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
                if (region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3]):
                    region_dets.append(det)
                    region_conf += det['combined_confidence']
            mean_conf = region_conf / len(region_dets) if region_dets else 0
            region_confidences.append({
                'detections': region_dets,
                'mean_confidence': mean_conf
            })
        return region_confidences

class ConfMixTrainer:
    def __init__(self, confmix_detector, source_data_path):
        self.confmix_detector = confmix_detector
        self.source_images_path = Path(source_data_path) / "images" / "train"
        self.source_labels_path = Path(source_data_path) / "labels" / "train"
        
        if not self.source_images_path.exists():
            raise FileNotFoundError(f"Source images path not found: {self.source_images_path}")
            
        self.source_images = list(self.source_images_path.glob("*.png"))
        print(f"Found {len(self.source_images)} CholecT50 source images")
    
    def _get_random_source_image(self):
        source_image_path = random.choice(self.source_images)
        source_image = Image.open(source_image_path)
        
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
        """Erstellt ein gemischtes Bild nach der ConfMix-Strategie."""
        source_data = self._get_random_source_image()
        source_image = source_data['image']
        
        if source_image.size != target_image.size:
            source_image = source_image.resize(target_image.size)
        
        best_region_idx = target_results['best_region_idx']
        best_region = target_results['regions'][best_region_idx]
        confidence = target_results['region_confidences'][best_region_idx]['mean_confidence']
        
        if confidence > CONFIDENCE_THRESHOLD:
            mask = np.zeros((target_image.size[1], target_image.size[0]))
            x1, y1, x2, y2 = best_region
            mask[y1:y2, x1:x2] = 1
            
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
    """Konvertiert Box-Koordinaten ins YOLO-Format."""
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height
    return x_center, y_center, width, height

def save_combined_labels(mixed_sample, label_path, image_width, image_height):
    """
    Speichert kombinierte Labels aus Source und Target (YOLO-Format).
    Dabei werden zunächst Target-Pseudo-Labels und anschließend Source-Labels ergänzt.
    Anschließend erfolgt die Filterung von Duplikaten mittels IoU.
    """
    target_region = mixed_sample['target_region']
    x1, y1, x2, y2 = target_region
    
    combined_labels = []
    
    # 1. Target-Pseudo-Labels (nur für die ausgewählte Region)
    for det in mixed_sample['target_detections']:
        box = det['box']
        box_center_x = (box[0] + box[2]) / 2
        box_center_y = (box[1] + box[3]) / 2
        
        if (x1 <= box_center_x <= x2 and y1 <= box_center_y <= y2):
            x_center, y_center, width, height = convert_to_yolo_format(box, image_width, image_height)
            combined_labels.append({
                'class': det['class'],
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
    
    # 2. Source-Labels (für den Rest des Bildes)
    for label in mixed_sample['source_data']['labels']:
        parts = label.strip().split()
        if len(parts) == 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            abs_x = x_center * image_width
            abs_y = y_center * image_height
            
            if not (x1 <= abs_x <= x2 and y1 <= abs_y <= y2):
                combined_labels.append({
                    'class': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
    
    # Filtere überlappende Labels (Duplikate) mittels IoU
    filtered_labels = filter_duplicate_labels(combined_labels, iou_threshold=IOU_THRESHOLD)
    
    with open(label_path, 'w') as f:
        for label in filtered_labels:
            f.write(f"{label['class']} {label['x_center']:.6f} {label['y_center']:.6f} "
                    f"{label['width']:.6f} {label['height']:.6f}\n")

# ---------------------------
# Funktionen zur Analyse & YAML-Erstellung
# ---------------------------
def analyze_class_distribution(labels_dir):
    """
    Analysiert die Klassenverteilung in allen YOLO-Label-Dateien.
    """
    class_counts = Counter()
    total_objects = 0
    
    for label_file in Path(labels_dir).rglob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    total_objects += 1
    return class_counts, total_objects

def calculate_class_weights(class_counts, total_objects, method='inverse'):
    """
    Berechnet Klassengewichte basierend auf der Verteilung.
    
    Methoden:
    - 'inverse': 1 / frequency (mit Normalisierung)
    - 'balanced': total_objects / (n_classes * count)
    """
    weights = {}
    n_classes = len(class_counts)
    
    if method == 'inverse':
        max_count = max(class_counts.values())
        for class_id, count in class_counts.items():
            weights[class_id] = max_count / (count + 1e-6)
    elif method == 'balanced':
        for class_id, count in class_counts.items():
            weights[class_id] = total_objects / (n_classes * count + 1e-6)
    
    max_weight = max(weights.values())
    for class_id in weights:
        weights[class_id] = weights[class_id] / max_weight
    
    return weights

def create_dataset_yaml(base_path, class_weights):
    """
    Erstellt (bzw. überschreibt) eine dataset.yaml im Basisordner,
    die unter anderem das Verzeichnislayout (train/val/test) und die berechneten Klassengewichte enthält.
    """
    data = {
        'train': os.path.join(base_path, 'images', 'train'),
        'val': os.path.join(base_path, 'images', 'val'),
        'test': os.path.join(base_path, 'images', 'test'),
        'class_weights': {str(k): float(v) for k, v in class_weights.items()}
    }
    yaml_path = os.path.join(base_path, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"Created dataset.yaml at {yaml_path}")

# ---------------------------
# Main-Funktion: Erstellung des Datasets + YAML-Update
# ---------------------------
def main():
    """Hauptfunktion zur Erstellung des balancierten Mixed-Samples-Datasets mit Split und YAML-Erstellung."""
    # Erstelle Unterordner für train, val und test
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(BALANCED_MIXED_SAMPLES_PATH, "images", split), exist_ok=True)
        os.makedirs(os.path.join(BALANCED_MIXED_SAMPLES_PATH, "labels", split), exist_ok=True)
        os.makedirs(os.path.join(BALANCED_MIXED_SAMPLES_PATH, "labels", split, "meta"), exist_ok=True)
    
    # Balancing-Parameter
    class_counts_tracker = {cls: 0 for cls in TOOL_MAPPING.keys()}
    MAX_SAMPLES_PER_CLASS = 1000  # Maximale Samples pro Klasse
    
    # Initialisiere Detector und Trainer (wie im alten Code)
    confidence_detector = ConfidenceBasedDetector(YOLO_MODEL_PATH)
    confmix_detector = ConfMixDetector(confidence_detector)
    trainer = ConfMixTrainer(confmix_detector, CHOLECT50_PATH)
    
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
                # Bestimme die dominante Zielklasse innerhalb der gemixten Region
                dominant_class = None
                best_conf = 0
                x1, y1, x2, y2 = mixed_sample['target_region']
                for det in mixed_sample['target_detections']:
                    box = det['box']
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                        if det['combined_confidence'] > best_conf:
                            best_conf = det['combined_confidence']
                            dominant_class = det['class']
                
                if dominant_class is None:
                    continue  # Kein dominantes Instrument gefunden
                
                # Überspringe Sample, falls diese Klasse bereits ausreichend vertreten ist
                if class_counts_tracker[dominant_class] >= MAX_SAMPLES_PER_CLASS:
                    continue
                
                # Weisen den aktuellen Sample zufällig einem Split zu (Train 80%, Val 10%, Test 10%)
                r = random.random()
                if r < 0.8:
                    split = "train"
                elif r < 0.9:
                    split = "val"
                else:
                    split = "test"
                
                base_filename = f"mixed_{successful_mixes:06d}"
                image_path = os.path.join(BALANCED_MIXED_SAMPLES_PATH, "images", split, f"{base_filename}.png")
                label_path = os.path.join(BALANCED_MIXED_SAMPLES_PATH, "labels", split, f"{base_filename}.txt")
                meta_path = os.path.join(BALANCED_MIXED_SAMPLES_PATH, "labels", split, "meta", f"{base_filename}_meta.txt")
                
                # Speichere das gemischte Bild und die Labels
                mixed_sample['mixed_image'].save(image_path)
                image_width, image_height = mixed_sample['mixed_image'].size
                save_combined_labels(mixed_sample, label_path, image_width, image_height)
                
                with open(meta_path, 'w') as f:
                    f.write(f"Progress Ratio: {progress_ratio:.3f}\n")
                    f.write(f"Source: {mixed_sample['source_data']['path'].name}\n")
                    f.write(f"Target region: {mixed_sample['target_region']}\n")
                    f.write(f"Confidence: {mixed_sample['confidence']}\n")
                    f.write("Original detections:\n")
                    for det in mixed_sample['target_detections']:
                        f.write(f"{det['class']} {det['combined_confidence']} {det['box']}\n")
                
                class_counts_tracker[dominant_class] += 1
                successful_mixes += 1
                
                if successful_mixes % 100 == 0:
                    print(f"Created {successful_mixes} balanced mixed samples (Progress: {progress_ratio:.2%})")
    
    # Nach Abschluss der Datenerstellung: Analyse der Klassenverteilung und Erzeugung der dataset.yaml
    labels_dir = os.path.join(BALANCED_MIXED_SAMPLES_PATH, "labels")
    print("\nAnalyzing class distribution...")
    counts, total_objects = analyze_class_distribution(labels_dir)
    print("\nClass distribution:")
    for class_id, count in sorted(counts.items()):
        percentage = (count / total_objects) * 100 if total_objects > 0 else 0
        print(f"Class {class_id}: {count} objects ({percentage:.2f}%)")
    
    print("\nCalculating class weights...")
    class_weights = calculate_class_weights(counts, total_objects, method='inverse')
    print("\nNew class weights:")
    for class_id, weight in sorted(class_weights.items()):
        print(f"Class {class_id}: {weight:.3f}")
    
    create_dataset_yaml(BALANCED_MIXED_SAMPLES_PATH, class_weights)
    print("\nDataset creation complete.")

if __name__ == "__main__":
    main()
