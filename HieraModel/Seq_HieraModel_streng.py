import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import json
#from Verbmodel.models.strict_freq_model import SimpleVerbRecognition

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
    Berechnet Average Precision mit interpolation aller Punkte.
    """
    if len(y_true) == 0:
        return 0.0
    
    # Sortiere nach Konfidenz absteigend
    sort_idx = np.argsort(y_pred)[::-1]
    y_true = y_true[sort_idx]
    y_pred = y_pred[sort_idx]
    
    # Berechne kumulative Metriken
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    
    # Berechne Precision und Recall
    precision = tp / (tp + fp)
    recall = tp / np.sum(y_true)
    
    # Füge Start- und Endpunkte hinzu
    precision = np.concatenate([[0], precision, [0]])
    recall = np.concatenate([[0], recall, [1]])
    
    # Interpoliere Precision
    for i in range(len(precision)-2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])
    
    # Berechne AP
    ap = 0
    for i in range(len(recall)-1):
        ap += (recall[i+1] - recall[i]) * precision[i+1]
    
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
        #model = SimpleVerbRecognition.load_from_checkpoint(model_path)
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
    
    def evaluate_frame(self, img_path, ground_truth, save_visualization=True):
        img = Image.open(img_path)
        original_img = img.copy()
        
        frame_predictions = {
            'instruments': defaultdict(list),
            'verbs': defaultdict(list),
            'pairs': defaultdict(list)
        }
        
        frame_metrics = {
            'instruments': {'TP': 0, 'FP': 0, 'FN': 0},
            'verbs': {'TP': 0, 'FP': 0, 'FN': 0},
            'pairs': {'TP': 0, 'FP': 0, 'FN': 0}
        }
        
        # YOLO Predictions
        yolo_results = self.yolo_model(img)
        pred_classes = yolo_results[0].boxes.cls.cpu().numpy()
        pred_confidences = yolo_results[0].boxes.conf.cpu().numpy()
        pred_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        
        # Erstelle ein Bild für Visualisierung
        draw = ImageDraw.Draw(original_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Liste für alle Vorhersagen (für spätere Anzeige)
        predictions_list = []
        
        # Verarbeite Detektionen
        for i in range(len(pred_classes)):
            instrument_class = int(pred_classes[i])
            confidence = float(pred_confidences[i])
            
            # Niedrigerer Threshold für Instrumentenerkennung
            INSTRUMENT_THRESHOLD = 0.3  # Du kannst diesen Wert anpassen
            
            if instrument_class < 6 and confidence >= INSTRUMENT_THRESHOLD:
                box = pred_boxes[i]
                predicted_instrument = TOOL_MAPPING[instrument_class]
                
                # Instrument Detection
                frame_predictions['instruments'][predicted_instrument].append(confidence)
                
                # Zeichne Box und Instrument-Label
                x1, y1, x2, y2 = map(int, box)
                draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                
                try:
                    crop = img.crop((x1, y1, x2, y2))
                    crop = self.transform(crop).unsqueeze(0).to(self.device)
                    
                    # Verb Prediction
                    with torch.no_grad():
                        verb_outputs = self.verb_model(crop, [predicted_instrument])
                        verb_probs = torch.nn.functional.softmax(verb_outputs, dim=1)
                        
                        # Finde das Verb mit der höchsten Wahrscheinlichkeit
                        max_verb_idx = torch.argmax(verb_probs).item()
                        max_verb_prob = float(verb_probs[0][max_verb_idx])
                        
                        # Niedrigerer Threshold für Verberkennung oder komplett ohne
                        VERB_THRESHOLD = 0.1 
                        
                        
                        verb_name = VERB_MAPPING[max_verb_idx]
                        frame_predictions['verbs'][verb_name].append(max_verb_prob)
                        
                        # Speichere die Vorhersage mit Konfidenzen
                        prediction_text = f"{predicted_instrument}-{verb_name}"
                        predictions_list.append((prediction_text, confidence, max_verb_prob))
                        
                        # Zeichne das Instrument-Verb-Paar über der Box
                        # Farbcodierung basierend auf Konfidenzen
                        if confidence >= CONFIDENCE_THRESHOLD and max_verb_prob >= CONFIDENCE_THRESHOLD:
                            text_color = 'blue'  # Hohe Konfidenz für beide
                        elif confidence >= INSTRUMENT_THRESHOLD:
                            text_color = 'orange'  # Mittlere Konfidenz
                        else:
                            text_color = 'red'  # Niedrige Konfidenz
                        
                        draw.text((x1, y1-25), 
                                f"{prediction_text}\nI:{confidence:.2f}, V:{max_verb_prob:.2f}", 
                                fill=text_color, font=font)
                        
                        # Optional: Zeige die Top-3 Verben mit ihren Wahrscheinlichkeiten
                        top3_verb_indices = torch.topk(verb_probs[0], k=3).indices.cpu().numpy()
                        top3_verb_probs = torch.topk(verb_probs[0], k=3).values.cpu().numpy()
                        verb_text = ""
                        for idx, prob in zip(top3_verb_indices, top3_verb_probs):
                            verb_text += f"{VERB_MAPPING[idx]}: {prob:.2f}\n"
                        draw.text((x1, y2+5), verb_text, fill=text_color, font=font)
                
                except Exception as e:
                    print(f"Error processing detection: {str(e)} for instrument {predicted_instrument}")
                    continue
        
        # Zeichne Ground Truth unten rechts
        img_width, img_height = img.size
        gt_text = "Ground Truth:"
        draw.text((img_width-300, img_height-150), gt_text, fill='green', font=font)
        
        # Zeige alle Ground Truth Instrument-Verb-Paare
        y_offset = img_height - 120
        for category, items in ground_truth.items():
            if category == 'pairs':  # Wir interessieren uns nur für die Paare
                for pair, count in items.items():
                    if count > 0:  # Wenn das Paar in diesem Frame existiert
                        draw.text((img_width-300, y_offset), f"- {pair}", fill='green', font=font)
                        y_offset += 25
        
        # Speichere das annotierte Bild
        if save_visualization:
            save_dir = "/data/Bartscht/VID92_val"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, os.path.basename(img_path))
            original_img.save(save_path)
        
        return frame_predictions, frame_metrics

    def evaluate(self):
        metrics = {
            'instruments': defaultdict(list),
            'verbs': defaultdict(list),
            'pairs': defaultdict(list)
        }
        
        total_metrics = {
            'instruments': {'TP': 0, 'FP': 0, 'FN': 0},
            'verbs': {'TP': 0, 'FP': 0, 'FN': 0},
            'pairs': {'TP': 0, 'FP': 0, 'FN': 0}
        }
        
        # Nur VID92 analysieren
        video = "VID92"
        print(f"\nProcessing {video}...")
        ground_truth = self.load_ground_truth(video)
        
        video_folder = os.path.join(self.dataset_dir, "CholecT50", "videos", video)
        frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.png')])
        
        for frame_file in tqdm(frame_files):
            frame_number = int(frame_file.split('.')[0])
            img_path = os.path.join(video_folder, frame_file)
            
            # Hole beide Rückgabewerte von evaluate_frame mit Visualisierung
            frame_predictions, frame_metrics = self.evaluate_frame(
                img_path,
                ground_truth[frame_number],
                save_visualization=True
            )
            
            # Aktualisiere die Gesamt-Metriken
            for category in ['instruments', 'verbs', 'pairs']:
                total_metrics[category]['TP'] += frame_metrics[category]['TP']
                total_metrics[category]['FP'] += frame_metrics[category]['FP']
                total_metrics[category]['FN'] += frame_metrics[category]['FN']
            
            # Überprüfe ob Detektionen in irgendeiner Kategorie vorhanden sind
            has_predictions = any(bool(preds) for category_preds in frame_predictions.values() 
                                for preds in category_preds.values())
            
            if not has_predictions:
                # Füge negative Samples für alle vorhandenen Ground Truth Annotationen hinzu
                for category in ['instruments', 'verbs', 'pairs']:
                    for item, count in ground_truth[frame_number][category].items():
                        if count > 0:
                            metrics[category][item].append({
                                'gt': True,
                                'pred_confidence': 0.0  # Konfidenz 0 für nicht erkannte Objekte
                            })
                            print(f"Missed {category}: {item} in frame {frame_number}")
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
                            
                            # Logging für False Positives
                            if not gt_count > 0:
                                print(f"False Positive {category}: {item} in frame {frame_number}")
                                metrics[category][item].append({
                                    'gt': False,
                                    'pred_confidence': pred_confidence
                                })
        
        # Berechne finale APs und drucke detaillierte Metriken
        results = {}
        print("\nFinal Evaluation Results for VID92:")
        print("================================")
        
        for category in ['instruments', 'verbs', 'pairs']:
            print(f"\n{category.upper()} METRICS:")
            print("-" * 20)
            print(f"Total True Positives: {total_metrics[category]['TP']}")
            print(f"Total False Positives: {total_metrics[category]['FP']}")
            print(f"Total False Negatives: {total_metrics[category]['FN']}")
            
            if total_metrics[category]['TP'] + total_metrics[category]['FP'] > 0:
                precision = total_metrics[category]['TP'] / (total_metrics[category]['TP'] + total_metrics[category]['FP'])
                recall = total_metrics[category]['TP'] / (total_metrics[category]['TP'] + total_metrics[category]['FN'])
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"Overall Precision: {precision:.4f}")
                print(f"Overall Recall: {recall:.4f}")
                print(f"Overall F1-Score: {f1:.4f}")
            
            category_aps = {}
            for item, predictions in metrics[category].items():
                if predictions:
                    y_true = np.array([p['gt'] for p in predictions])
                    y_pred = np.array([p['pred_confidence'] for p in predictions])
                    ap = calculate_ap(y_true, y_pred)
                    category_aps[item] = ap
                    
                    print(f"\n{item}:")
                    print(f"AP: {ap:.4f}")
                    print(f"Total predictions: {len(predictions)}")
                    print(f"True positives: {np.sum(y_true)}")
                    print(f"False positives: {len(predictions) - np.sum(y_true)}")
                    
                    # Zusätzliche Statistiken
                    if len(predictions) > 0:
                        mean_conf = np.mean([p['pred_confidence'] for p in predictions])
                        print(f"Mean confidence: {mean_conf:.4f}")
            
            mean_ap = np.mean(list(category_aps.values())) if category_aps else 0
            results[category] = {
                'per_class_ap': category_aps,
                'mAP': mean_ap,
                'metrics': total_metrics[category]
            }
            print(f"\nmAP: {mean_ap:.4f}")
        
        # Speichere die Ergebnisse in einer JSON-Datei
        results_path = os.path.join("/data/Bartscht/VID92_val", "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'metrics': total_metrics,
                'results': {k: {
                    'mAP': v['mAP'],
                    'per_class_ap': v['per_class_ap'],
                    'metrics': v['metrics']
                } for k, v in results.items()}
            }, f, indent=4)
        
        print(f"\nDetailed results saved to: {results_path}")
        return results

if __name__ == "__main__":
    # Definiere die Pfade
    base_dir = "/home/Bartscht/YOLO"
    yolo_path = os.path.join(base_dir, "runs/detect/train35/weights/best.pt")
    verb_model_path = os.path.join(base_dir, "Verbmodel/checkpoints/last-v2.ckpt")
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