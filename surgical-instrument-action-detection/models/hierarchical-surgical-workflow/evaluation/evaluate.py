import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import json
from tqdm import tqdm
from ultralytics import YOLO

# Setze Working Directory und Python Path für korrekte Imports
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
verb_recognition_dir = current_dir.parent / 'verb_recognition'
os.chdir(verb_recognition_dir)
sys.path.insert(0, str(verb_recognition_dir))

from models.SurgicalActionNet import SurgicalVerbRecognition

# Globale Konfiguration
class Config:
    CONFIDENCE_THRESHOLD = 0.6
    IOU_THRESHOLD = 0.3
    VIDEOS_TO_ANALYZE = ["VID92", "VID96", "VID103", "VID110", "VID111"]
    
    # Paths
    BASE_DIR = Path("/home/Bartscht/YOLO")
    DATASET_DIR = Path("/data/Bartscht")
    YOLO_WEIGHTS = BASE_DIR / "runs/detect/train35/weights/best.pt"
    VERB_WEIGHTS = BASE_DIR / "surgical-instrument-action-detection/models/hierarchical-surgical-workflow/verb_recognition/checkpoints/expert-field/expert-field-epoch33/loss=0.824.ckpt"
    OUTPUT_DIR = Path("/data/Bartscht/VID92_val")
    
    # Mappings
    TOOL_MAPPING = {
        0: 'grasper',
        1: 'bipolar',
        2: 'hook',
        3: 'scissors',
        4: 'clipper',
        5: 'irrigator'
    }

    VERB_MAPPING = {
        0: 'grasp',
        1: 'retract',
        2: 'dissect',
        3: 'coagulate',
        4: 'clip',
        5: 'cut',
        6: 'aspirate',
        7: 'irrigate',
        8: 'pack',
        9: 'null_verb'
    }
    
    # Image transforms
    TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

class HierarchicalEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Modelle laden
        self.load_models()
        
        # Output Directory erstellen
        self.config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    def load_models(self):
        """Lädt beide Modelle (YOLO und VerbRecognition)"""
        try:
            # YOLO laden
            print("Loading YOLO model...")
            self.yolo_model = YOLO(str(self.config.YOLO_WEIGHTS))
            print("YOLO model loaded successfully")
            
            # Verb Recognition Model laden
            print("Loading Verb Recognition model...")
            self.verb_model = SurgicalVerbRecognition.load_from_checkpoint(
                str(self.config.VERB_WEIGHTS),
                map_location=self.device
            )
            self.verb_model.to(self.device)
            self.verb_model.eval()
            print("Verb Recognition model loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
    
    def load_ground_truth(self, video):
        """Lädt Ground Truth Annotationen"""
        labels_folder = self.config.DATASET_DIR / "CholecT50" / "labels"
        json_file = labels_folder / f"{video}.json"
        
        frame_annotations = defaultdict(lambda: {
            'instruments': defaultdict(int),
            'verbs': defaultdict(int),
            'pairs': defaultdict(int)
        })
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            annotations = data['annotations']
            
            for frame, instances in annotations.items():
                frame_number = int(frame)
                for instance in instances:
                    instrument = instance[1]
                    verb = instance[7]
                    
                    if isinstance(instrument, int) and 0 <= instrument < 6:
                        instrument_name = self.config.TOOL_MAPPING[instrument]
                        frame_annotations[frame_number]['instruments'][instrument_name] += 1
                        
                        if isinstance(verb, int) and 0 <= verb < 10:
                            verb_name = self.config.VERB_MAPPING[verb]
                            frame_annotations[frame_number]['verbs'][verb_name] += 1
                            pair_key = f"{instrument_name}_{verb_name}"
                            frame_annotations[frame_number]['pairs'][pair_key] += 1
        
        return frame_annotations

    def process_frame(self, img_path, ground_truth):
        """Verarbeitet ein einzelnes Frame"""
        img = Image.open(img_path)
        original_img = img.copy()
        
        # Vorbereitung der Metriken
        frame_predictions = {
            'instruments': defaultdict(list),
            'verbs': defaultdict(list),
            'pairs': defaultdict(list)
        }
        
        # YOLO Predictions
        yolo_results = self.yolo_model(img)
        
        # Visualization vorbereiten
        draw = ImageDraw.Draw(original_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # YOLO Detektionen verarbeiten
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                # Box Information extrahieren
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                if confidence >= self.config.CONFIDENCE_THRESHOLD:
                    instrument_name = self.config.TOOL_MAPPING[class_id]
                    
                    # Crop und Transform
                    crop = img.crop((x1, y1, x2, y2))
                    crop_tensor = self.config.TRANSFORMS(crop).unsqueeze(0).to(self.device)
                    
                    # Verb Prediction
                    with torch.no_grad():
                        verb_output = self.verb_model(crop_tensor)
                        verb_probs = torch.nn.functional.softmax(verb_output, dim=1)
                        max_verb_idx = torch.argmax(verb_probs).item()
                        verb_name = self.config.VERB_MAPPING[max_verb_idx]
                        verb_conf = verb_probs[0][max_verb_idx].item()
                    
                    # Speichere Predictions
                    frame_predictions['instruments'][instrument_name].append(confidence)
                    frame_predictions['verbs'][verb_name].append(verb_conf)
                    frame_predictions['pairs'][f"{instrument_name}_{verb_name}"].append(
                        min(confidence, verb_conf)
                    )
                    
                    # Visualisierung
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                    label = f"{instrument_name}-{verb_name}\nI:{confidence:.2f}, V:{verb_conf:.2f}"
                    draw.text((x1, y1-25), label, fill='blue', font=font)
        
        # Ground Truth visualisieren
        self._draw_ground_truth(draw, ground_truth, img.size, font)
        
        # Speichere annotiertes Bild
        save_path = self.config.OUTPUT_DIR / os.path.basename(img_path)
        original_img.save(save_path)
        
        return frame_predictions
    
    def _draw_ground_truth(self, draw, ground_truth, img_size, font):
        """Zeichnet Ground Truth Informationen"""
        img_width, img_height = img_size
        y_offset = img_height - 150
        
        draw.text((img_width-300, y_offset), "Ground Truth:", fill='green', font=font)
        y_offset += 25
        
        for pair, count in ground_truth['pairs'].items():
            if count > 0:
                draw.text((img_width-300, y_offset), f"- {pair}", fill='green', font=font)
                y_offset += 25

    def evaluate(self):
        """Führt die komplette Evaluation durch"""
        results = {}
        
        for video in self.config.VIDEOS_TO_ANALYZE:
            print(f"\nProcessing {video}...")
            
            # Ground Truth laden
            ground_truth = self.load_ground_truth(video)
            
            # Frames laden
            video_folder = self.config.DATASET_DIR / "CholecT50" / "videos" / video
            frame_files = sorted(
                [f for f in os.listdir(video_folder) if f.endswith('.png')],
                key=lambda x: int(x.split('.')[0])
            )
            
            video_metrics = defaultdict(list)
            
            # Frames verarbeiten
            for frame_file in tqdm(frame_files, desc=f"Processing {video}"):
                frame_number = int(frame_file.split('.')[0])
                img_path = video_folder / frame_file
                
                # Frame verarbeiten
                frame_predictions = self.process_frame(
                    str(img_path),
                    ground_truth[frame_number]
                )
                
                # Metriken sammeln
                self._update_metrics(video_metrics, frame_predictions, 
                                  ground_truth[frame_number])
            
            # Ergebnisse berechnen
            results[video] = self._calculate_metrics(video_metrics)
            
            # Ergebnisse speichern
            metrics_file = self.config.OUTPUT_DIR / f"{video}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(results[video], f, indent=4)
            
            # Ergebnisse ausgeben
            self._print_metrics(video, results[video])
        
        return results

    def _update_metrics(self, metrics, predictions, ground_truth):
        """Aktualisiert die Evaluationsmetriken"""
        for category in ['instruments', 'verbs', 'pairs']:
            for item, predictions_list in predictions[category].items():
                if predictions_list:  # Wenn Vorhersagen existieren
                    metrics[category].append({
                        'item': item,
                        'prediction': max(predictions_list),  # Beste Konfidenz
                        'ground_truth': ground_truth[category][item] > 0
                    })
    
    def _calculate_metrics(self, metrics):
        """Berechnet die finalen Metriken"""
        results = {}
        
        for category in ['instruments', 'verbs', 'pairs']:
            if not metrics[category]:
                continue
                
            y_true = np.array([m['ground_truth'] for m in metrics[category]])
            y_pred = np.array([m['prediction'] for m in metrics[category]])
            
            results[category] = {
                'precision': self._calculate_precision(y_true, y_pred),
                'recall': self._calculate_recall(y_true, y_pred),
                'f1_score': self._calculate_f1(y_true, y_pred),
                'accuracy': self._calculate_accuracy(y_true, y_pred)
            }
        
        return results
    
    def _calculate_precision(self, y_true, y_pred, threshold=0.5):
        """Berechnet Precision"""
        predictions = (y_pred >= threshold).astype(int)
        tp = np.sum((predictions == 1) & (y_true == 1))
        fp = np.sum((predictions == 1) & (y_true == 0))
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    def _calculate_recall(self, y_true, y_pred, threshold=0.5):
        """Berechnet Recall"""
        predictions = (y_pred >= threshold).astype(int)
        tp = np.sum((predictions == 1) & (y_true == 1))
        fn = np.sum((predictions == 0) & (y_true == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    def _calculate_f1(self, y_true, y_pred, threshold=0.5):
        """Berechnet F1-Score"""
        precision = self._calculate_precision(y_true, y_pred, threshold)
        recall = self._calculate_recall(y_true, y_pred, threshold)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    def _calculate_accuracy(self, y_true, y_pred, threshold=0.5):
        """Berechnet Accuracy"""
        predictions = (y_pred >= threshold).astype(int)
        return np.mean(predictions == y_true)
    
    def _print_metrics(self, video, results):
        """Gibt die Metriken aus"""
        print(f"\nResults for {video}:")
        print("=" * 50)
        
        for category, metrics in results.items():
            print(f"\n{category.upper()}:")
            print("-" * 30)
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

def main():
    """Hauptfunktion für die Evaluation"""
    try:
        print("\n=== Starting Surgical Workflow Evaluation ===")
        
        # Konfiguration erstellen
        config = Config()
        
        # Prüfe ob alle notwendigen Pfade existieren
        required_paths = [
            config.YOLO_WEIGHTS,
            config.VERB_WEIGHTS,
            config.DATASET_DIR
        ]
        
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required path not found: {path}")
        
        print("\nAll required paths verified.")
        print(f"YOLO weights: {config.YOLO_WEIGHTS}")
        print(f"Verb model weights: {config.VERB_WEIGHTS}")
        print(f"Dataset directory: {config.DATASET_DIR}")
        
        # Erstelle Output-Verzeichnis
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory created/verified: {config.OUTPUT_DIR}")
        
        # Evaluator initialisieren
        print("\nInitializing evaluator...")
        evaluator = HierarchicalEvaluator(config)
        
        # Evaluation durchführen
        print("\nStarting evaluation...")
        results = evaluator.evaluate()
        
        # Gesamtergebnisse speichern
        results_file = config.OUTPUT_DIR / "complete_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Zusammenfassung der Ergebnisse
        print("\n=== Evaluation Summary ===")
        for video, video_results in results.items():
            print(f"\nResults for {video}:")
            print("-" * 30)
            
            for category, metrics in video_results.items():
                print(f"\n{category.upper()}:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
        
        print(f"\nComplete results saved to: {results_file}")
        print("\nEvaluation completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\nError: Required file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()