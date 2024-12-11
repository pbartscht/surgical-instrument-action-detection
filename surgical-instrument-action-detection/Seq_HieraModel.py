import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from collections import defaultdict
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import json
#from Verbmodel.models.model import SimpleVerbRecognition

CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.3
VIDEOS_TO_ANALYZE = ["VID92", "VID96", "VID103", "VID110", "VID111"]

# Global mappings
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

def non_max_suppression(boxes, scores, iou_threshold):
    """
    Führt Non-Maximum Suppression für mehrere Boxen durch.
    
    Args:
        boxes (np.ndarray): Array der Bounding Boxes im Format (N, 4)
        scores (np.ndarray): Array der Konfidenzwerte im Format (N,)
        iou_threshold (float): IoU Schwellenwert für NMS
    
    Returns:
        np.ndarray: Indizes der beibehaltenen Boxen
    """
    # Konvertiere zu Torch Tensoren
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    
    # Stelle sicher, dass boxes die richtige Form hat
    if boxes.dim() == 1:
        boxes = boxes.view(1, 4)
    
    # Führe NMS durch
    keep = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    
    return keep.numpy()

def calculate_precision_recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> tuple[float, float, dict]:
    """
    Berechnet Precision und Recall für einen Schwellenwert.
    """
    predictions = (y_pred >= threshold).astype(int)
    TP = np.sum((predictions == 1) & (y_true == 1))
    FP = np.sum((predictions == 1) & (y_true == 0))
    FN = np.sum((predictions == 0) & (y_true == 1))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    return precision, recall, {'TP': TP, 'FP': FP, 'FN': FN}

def calculate_ap(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Berechnet AP basierend auf sortierten Konfidenzwerten.
    """
    sort_idx = np.argsort(y_pred)[::-1]
    y_true = y_true[sort_idx]
    y_pred = y_pred[sort_idx]
    
    precisions = []
    recalls = []
    
    tp_cumsum = np.cumsum(y_true)
    fp_cumsum = np.cumsum(1 - y_true)
    total_positives = np.sum(y_true)
    
    for i in range(len(y_true)):
        if total_positives > 0:
            precision = tp_cumsum[i] / (tp_cumsum[i] + fp_cumsum[i])
            recall = tp_cumsum[i] / total_positives
            precisions.append(precision)
            recalls.append(recall)
    
    if not precisions:
        return 0.0
        
    # Füge Start- und Endpunkte hinzu
    precisions = [0] + precisions + [0]
    recalls = [0] + recalls + [1]
    
    # Berechne AP
    ap = 0
    for i in range(len(recalls) - 1):
        ap += (recalls[i+1] - recalls[i]) * precisions[i+1]
    
    return ap

class HierarchicalEvaluator:
    def __init__(self, yolo_path, verb_model_path, dataset_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = YOLO(yolo_path)
        self.verb_model = self.load_verb_model(verb_model_path)
        self.dataset_dir = dataset_dir
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_verb_model(self, model_path):
      #  model = SimpleVerbRecognition.load_from_checkpoint(model_path)
        model = model.to(self.device)
        model.eval()
        return model
    
    def load_ground_truth(self, video):
        labels_folder = os.path.join(self.dataset_dir, "CholecT50", "labels")
        json_file = os.path.join(labels_folder, f"{video}.json")
        
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
                        instrument_name = TOOL_MAPPING[instrument]
                        frame_annotations[frame_number]['instruments'][instrument_name] += 1
                        
                        if isinstance(verb, int) and 0 <= verb < 10:
                            verb_name = VERB_MAPPING[verb]
                            frame_annotations[frame_number]['verbs'][verb_name] += 1
                            pair_key = f"{instrument_name}_{verb_name}"
                            frame_annotations[frame_number]['pairs'][pair_key] += 1
        
        return frame_annotations
    
    def evaluate_frame(self, img_path, ground_truth):
        img = Image.open(img_path)
        
        # Initialisiere frame_predictions immer
        frame_predictions = {
            'instruments': defaultdict(list),
            'verbs': defaultdict(list),
            'pairs': defaultdict(list)
        }
        
        # YOLO Predictions
        yolo_results = self.yolo_model(img)
        
        pred_classes = yolo_results[0].boxes.cls.cpu().numpy()
        pred_confidences = yolo_results[0].boxes.conf.cpu().numpy()
        pred_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        
        # Wenn keine Detektionen gefunden wurden, gebe leeres frame_predictions zurück
        if len(pred_classes) == 0:
            return frame_predictions
        
        # Validiere verfügbare Instrument-Namen
        valid_instruments = set(self.verb_model.instrument_classes.keys())
        
        # Verarbeite jede Detektion einzeln
        for i in range(len(pred_classes)):
            instrument_class = int(pred_classes[i])
            confidence = pred_confidences[i]
            
            if instrument_class < 6 and confidence >= CONFIDENCE_THRESHOLD:
                box = pred_boxes[i]
                instrument_name = TOOL_MAPPING[instrument_class]
                
                # Überprüfe ob das Instrument vom VerbModel unterstützt wird
                if instrument_name not in valid_instruments:
                    print(f"Warning: Instrument {instrument_name} not supported by verb model")
                    continue
                
                # Führe Non-Maximum Suppression nur für dieses spezifische Instrument durch
                overlapping_mask = pred_classes == instrument_class
                overlapping_boxes = pred_boxes[overlapping_mask]
                overlapping_confidences = pred_confidences[overlapping_mask]
                
                if len(overlapping_boxes) > 0:
                    # Korrekte Formatierung der Boxen für NMS
                    # Stelle sicher, dass die Boxen die richtige Form haben (N x 4)
                    keep = non_max_suppression(
                        overlapping_boxes, 
                        overlapping_confidences, 
                        IOU_THRESHOLD
                    )
                    
                    # Überprüfe ob die aktuelle Box nach NMS behalten wird
                    if i in keep:  # Wenn diese Box nach NMS behalten wird
                        # Instrument Detection
                        frame_predictions['instruments'][instrument_name].append(confidence)
                        
                        try:
                            # Crop und Verb-Prediction direkt im Anschluss
                            x1, y1, x2, y2 = map(int, box)
                            crop = img.crop((x1, y1, x2, y2))
                            crop = self.transform(crop).unsqueeze(0).to(self.device)
                            
                            # Verb Prediction für dieses spezifische Instrument
                            with torch.no_grad():
                                verb_outputs = self.verb_model(crop, [instrument_name])
                                verb_probs = torch.nn.functional.softmax(verb_outputs, dim=1)
                                
                                # Verarbeite Verb-Vorhersagen sofort
                                for verb_idx, prob in enumerate(verb_probs[0]):
                                    if prob >= CONFIDENCE_THRESHOLD:
                                        verb_name = VERB_MAPPING[verb_idx]
                                        frame_predictions['verbs'][verb_name].append(prob.item())
                                        
                                        # Erstelle Instrument-Verb Pair sofort
                                        pair_key = f"{instrument_name}_{verb_name}"
                                        frame_predictions['pairs'][pair_key].append(
                                            confidence * prob.item()
                                        )
                        
                        except Exception as e:
                            print(f"Error processing instrument {instrument_name}: {str(e)}")
                            continue
        
        return frame_predictions

    def evaluate(self):
        metrics = {
            'instruments': defaultdict(list),
            'verbs': defaultdict(list),
            'pairs': defaultdict(list)
        }
        
        for video in VIDEOS_TO_ANALYZE:
            print(f"\nProcessing {video}...")
            ground_truth = self.load_ground_truth(video)
            
            video_folder = os.path.join(self.dataset_dir, "CholecT50", "videos", video)
            frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.png')])
            
            for frame_file in tqdm(frame_files):
                frame_number = int(frame_file.split('.')[0])
                img_path = os.path.join(video_folder, frame_file)
                
                frame_predictions = self.evaluate_frame(img_path, ground_truth[frame_number])
                
                # Überprüfe ob Detektionen in irgendeiner Kategorie vorhanden sind
                has_predictions = any(bool(preds) for preds in frame_predictions.values())
                
                if not has_predictions:
                    # Füge negative Samples für alle vorhandenen Ground Truth Annotationen hinzu
                    for category in ['instruments', 'verbs', 'pairs']:
                        for item, count in ground_truth[frame_number][category].items():
                            if count > 0:
                                metrics[category][item].append({
                                    'gt': True,
                                    'pred_confidence': 0.0  # Konfidenz 0 für nicht erkannte Objekte
                                })
                else:
                    # Update metrics für gefundene Detektionen
                    for category in ['instruments', 'verbs', 'pairs']:
                        for item, predictions in frame_predictions[category].items():
                            if predictions:  # Wenn Vorhersagen existieren
                                gt_count = ground_truth[frame_number][category][item]
                                pred_confidence = max(predictions)  # Höchste Konfidenz
                                
                                metrics[category][item].append({
                                    'gt': gt_count > 0,
                                    'pred_confidence': pred_confidence
                                })
                                
                                # Füge auch False Positives hinzu
                                if not gt_count > 0:
                                    metrics[category][item].append({
                                        'gt': False,
                                        'pred_confidence': pred_confidence
                                    })
        
        # Berechne finale APs
        results = {}
        for category in ['instruments', 'verbs', 'pairs']:
            category_aps = {}
            total_predictions = 0
            total_ground_truth = 0
            
            for item, predictions in metrics[category].items():
                if predictions:
                    y_true = np.array([p['gt'] for p in predictions])
                    y_pred = np.array([p['pred_confidence'] for p in predictions])
                    
                    total_predictions += len(predictions)
                    total_ground_truth += np.sum(y_true)
                    
                    ap = calculate_ap(y_true, y_pred)
                    category_aps[item] = ap
                    
                    print(f"\n{category} - {item}:")
                    print(f"Total predictions: {len(predictions)}")
                    print(f"True positives: {np.sum(y_true)}")
                    print(f"AP: {ap:.4f}")
            
            mean_ap = np.mean(list(category_aps.values())) if category_aps else 0
            results[category] = {
                'per_class_ap': category_aps,
                'mAP': mean_ap,
                'total_predictions': total_predictions,
                'total_ground_truth': total_ground_truth
            }
        
        return results

if __name__ == "__main__":
    # Definiere die Pfade
    base_dir = "/home/Bartscht/YOLO"
    yolo_path = os.path.join(base_dir, "runs/detect/train35/weights/best.pt")
    verb_model_path = os.path.join(base_dir, "Verbmodel/checkpoints/last-v1.ckpt")
    dataset_dir = "/data/Bartscht"
    
    print(f"Checking paths:")
    print(f"YOLO model path: {yolo_path}")
    print(f"Verb model path: {verb_model_path}")
    print(f"Dataset directory: {dataset_dir}")
    
    # Überprüfe ob die Dateien existieren
    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"YOLO model not found at: {yolo_path}")
    if not os.path.exists(verb_model_path):
        raise FileNotFoundError(f"Verb model not found at: {verb_model_path}")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found at: {dataset_dir}")
    
    print("\nInitializing evaluator...")
    evaluator = HierarchicalEvaluator(
        yolo_path=yolo_path,
        verb_model_path=verb_model_path,
        dataset_dir=dataset_dir
    )
    
    print("\nStarting evaluation...")
    results = evaluator.evaluate()
    
    # Print results
    for category in ['instruments', 'verbs', 'pairs']:
        print(f"\n{category.upper()} RESULTS:")
        print("-" * 20)
        for item, ap in results[category]['per_class_ap'].items():
            print(f"{item}: {ap:.4f}")
        print(f"mAP: {results[category]['mAP']:.4f}")