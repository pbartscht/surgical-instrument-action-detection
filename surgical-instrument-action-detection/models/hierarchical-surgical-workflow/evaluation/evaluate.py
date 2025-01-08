import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO
import pytorch_lightning as pl

# Path configuration
current_dir = Path(__file__).resolve().parent
hierarchical_dir = current_dir.parent
sys.path.append(str(hierarchical_dir))

# Custom imports
from verb_recognition.models.SurgicalActionNet import SurgicalVerbRecognition

# Constants
CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.3
VIDEOS_TO_ANALYZE = ["VID92"]  # Initially only VID92

# Global mappings
TOOL_MAPPING = {
    0: 'grasper', 1: 'bipolar', 2: 'hook', 
    3: 'scissors', 4: 'clipper', 5: 'irrigator'
}

VERB_MAPPING = {
    0: 'grasp', 1: 'retract', 2: 'dissect', 3: 'coagulate', 
    4: 'clip', 5: 'cut', 6: 'aspirate', 7: 'irrigate', 
    8: 'pack', 9: 'null_verb'
}

# Mapping between verb model indices and evaluation indices
VERB_MODEL_TO_EVAL_MAPPING = {
    0: 2,   # Model: 'dissect' -> Eval: 'dissect'
    1: 1,   # Model: 'retract' -> Eval: 'retract'
    2: 9,   # Model: 'null_verb' -> Eval: 'null_verb'
    3: 3,   # Model: 'coagulate' -> Eval: 'coagulate'
    4: 0,   # Model: 'grasp' -> Eval: 'grasp'
    5: 4,   # Model: 'clip' -> Eval: 'clip'
    6: 6,   # Model: 'aspirate' -> Eval: 'aspirate'
    7: 5,   # Model: 'cut' -> Eval: 'cut'
    8: 7,   # Model: 'irrigate' -> Eval: 'irrigate'
    9: 9    # Model: 'null_verb' -> Eval: 'null_verb'
}

def calculate_precision_recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> tuple[float, float, dict]:
    """Calculates precision and recall for a threshold."""
    predictions = (y_pred >= threshold).astype(int)
    TP = np.sum((predictions == 1) & (y_true == 1))
    FP = np.sum((predictions == 1) & (y_true == 0))
    FN = np.sum((predictions == 0) & (y_true == 1))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    return precision, recall, {'TP': TP, 'FP': FP, 'FN': FN}

def calculate_ap(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Average Precision with interpolation of all points."""
    if len(y_true) == 0:
        return 0.0
    
    # Sort by confidence in descending order
    sort_idx = np.argsort(y_pred)[::-1]
    y_true = y_true[sort_idx]
    y_pred = y_pred[sort_idx]
    
    # Calculate cumulative metrics
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    
    # Calculate precision and recall
    precision = tp / (tp + fp)
    recall = tp / np.sum(y_true)
    
    # Add start and end points
    precision = np.concatenate([[0], precision, [0]])
    recall = np.concatenate([[0], recall, [1]])
    
    # Interpolate precision
    for i in range(len(precision)-2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])
    
    # Calculate AP
    ap = 0
    for i in range(len(recall)-1):
        ap += (recall[i+1] - recall[i]) * precision[i+1]
    
    return ap

class ModelLoader:
    def __init__(self):
        self.hierarchical_dir = hierarchical_dir
        self.setup_paths()

    def setup_paths(self):
        """Defines all important paths for the models"""
        # YOLO model path
        self.yolo_weights = self.hierarchical_dir / "Instrument-classification-detection/weights/instrument_detector/best_v35.pt"
        # Verb model path
        self.verb_model_path = self.hierarchical_dir / "verb_recognition/checkpoints/expert-field/expert-field-epoch33/loss=0.824.ckpt"
        
        # Dataset path
        self.dataset_path = Path("/data/Bartscht/CholecT50")
        
        print(f"YOLO weights path: {self.yolo_weights}")
        print(f"Verb model path: {self.verb_model_path}")
        print(f"Dataset path: {self.dataset_path}")

        # Validate paths
        if not self.yolo_weights.exists():
            raise FileNotFoundError(f"YOLO weights not found at: {self.yolo_weights}")
        if not self.verb_model_path.exists():
            raise FileNotFoundError(f"Verb model checkpoint not found at: {self.verb_model_path}")
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.dataset_path}")

    def load_yolo_model(self):
        try:
            model = YOLO(str(self.yolo_weights))
            print("YOLO model loaded successfully")
            return model
        except Exception as e:
            print(f"Error details: {str(e)}")
            raise Exception(f"Error loading YOLO model: {str(e)}")

    def load_verb_model(self):
        try:
            model = SurgicalVerbRecognition.load_from_checkpoint(
                checkpoint_path=str(self.verb_model_path)
            )
            model.eval()
            print("Verb recognition model loaded successfully")
            return model
        except Exception as e:
            print(f"Error details: {str(e)}")
            raise Exception(f"Error loading verb model: {str(e)}")
        
class HierarchicalEvaluator:
    def __init__(self, yolo_model, verb_model, dataset_dir):
        """
        Initializes the HierarchicalEvaluator.
        
        :param yolo_model: Pre-trained YOLO model for instrument detection
        :param verb_model: Pre-trained verb recognition model
        :param dataset_dir: Directory of the CholecT50 dataset
        """
        # Device for computations (CUDA if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # YOLO model for instrument detection
        self.yolo_model = yolo_model
        
        # Move verb recognition model to computing device
        self.verb_model = verb_model.to(self.device)
        self.verb_model.eval()  # Set model to evaluation mode
        
        # Path to dataset
        self.dataset_dir = dataset_dir
        
        # Transformations for image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Size change
            transforms.ToTensor(),  # Convert to tensor
            # Normalization with ImageNet means and standard deviations
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.VALID_PAIRS = {
            'grasper': ['grasp', 'retract', 'null_verb'],
            'hook': ['dissect', 'cut', 'null_verb', 'coagulate'],
            'bipolar': ['coagulate', 'dissect', 'null_verb'],
            'clipper': ['clip', 'null_verb'],
            'scissors': ['cut', 'null_verb'],
            'irrigator': ['aspirate', 'irrigate', 'null_verb']
        }

    def load_ground_truth(self, video):
        """
        Loads ground truth annotations for a specific video.
        
        :param video: Video identifier (e.g., "VID92")
        :return: Dictionary with frame annotations
        """
        # Path to label files
        labels_folder = os.path.join(self.dataset_dir, "labels")
        json_file = os.path.join(labels_folder, f"{video}.json")
        
        # Defaultdict for frame annotations
        frame_annotations = defaultdict(lambda: {
            'instruments': defaultdict(int),
            'verbs': defaultdict(int),
            'pairs': defaultdict(int)
        })
        
        # Load JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)
            annotations = data['annotations']
            
            # Process annotations
            for frame, instances in annotations.items():
                frame_number = int(frame)
                for instance in instances:
                    instrument = instance[1]
                    verb = instance[7]
                    
                    # Validate and map instrument
                    if isinstance(instrument, int) and 0 <= instrument < 6:
                        instrument_name = TOOL_MAPPING[instrument]
                        frame_annotations[frame_number]['instruments'][instrument_name] += 1
                        
                        # Validate and map verb
                        if isinstance(verb, int) and 0 <= verb < 10:
                            verb_name = VERB_MAPPING[verb]
                            frame_annotations[frame_number]['verbs'][verb_name] += 1
                            
                            # Create instrument-verb pair
                            pair_key = f"{instrument_name}_{verb_name}"
                            frame_annotations[frame_number]['pairs'][pair_key] += 1
        
        return frame_annotations

    def evaluate_frame(self, img_path, ground_truth, save_visualization=True):
        """
        Evaluates a frame according to the hierarchical recognition process:
        1. YOLO detects instruments
        2. Each instrument is processed sequentially
        3. Verb prediction with valid pairs
        
        :param img_path: Path to image
        :param ground_truth: Ground truth annotations for the frame
        :param save_visualization: Whether visualization should be saved
        :return: List of recognized instrument-verb pairs
        """
        # Load and prepare image
        img = Image.open(img_path)
        original_img = img.copy()
        draw = ImageDraw.Draw(original_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        try:
            yolo_results = self.yolo_model(img)
            valid_detections = []
            
            # Print all detected instruments first
            print("\nDetected instruments in frame:")
            for detection in yolo_results[0].boxes:
                instrument_class = int(detection.cls)
                confidence = float(detection.conf)
                if instrument_class < 6 and confidence >= CONFIDENCE_THRESHOLD:
                    instrument_name = TOOL_MAPPING[instrument_class]
                    print(f"- Found {instrument_name} with confidence {confidence:.2f}")
                    valid_detections.append({
                        'class': instrument_class,
                        'confidence': confidence,
                        'box': detection.xyxy[0]
                    })
            
            valid_detections.sort(key=lambda x: x['confidence'], reverse=True)
            frame_pairs = []
            
            # Process each detection
            for idx, detection in enumerate(valid_detections):
                print(f"\nProcessing instrument {idx + 1}:")
                instrument_class = detection['class']
                instrument_name = TOOL_MAPPING[instrument_class]
                box = detection['box']
                confidence = detection['confidence']
                
                print(f"- Working on {instrument_name} (confidence: {confidence:.2f})")
                
                x1, y1, x2, y2 = map(int, box)
                instrument_crop = img.crop((x1, y1, x2, y2))
                crop_tensor = self.transform(instrument_crop).unsqueeze(0).to(self.device)
                
                # Get verb predictions
                verb_outputs = self.verb_model(crop_tensor, [instrument_name])
                verb_probs = verb_outputs['probabilities']
                
                # Print top 3 verb predictions for this instrument
                print(f"Top 3 verb predictions for {instrument_name}:")
                top_verbs = []
                for verb_model_idx in torch.topk(verb_probs[0], k=3).indices.cpu().numpy():
                    try:
                        eval_verb_idx = VERB_MODEL_TO_EVAL_MAPPING[verb_model_idx]
                        verb_name = VERB_MAPPING[eval_verb_idx]
                        verb_prob = float(verb_probs[0][verb_model_idx])
                        
                        print(f"  - {verb_name}: {verb_prob:.3f}")
                        
                        if (verb_name in self.VALID_PAIRS[instrument_name] and 
                            verb_prob > 0 and verb_name != 'null_verb'):
                            top_verbs.append({
                                'name': verb_name,
                                'probability': verb_prob
                            })
                    except KeyError as e:
                        print(f"Warning: Unexpected verb model index {verb_model_idx}")
                
                # Best verb selection and visualization
                if top_verbs:
                    best_verb = max(top_verbs, key=lambda x: x['probability'])
                    pair = f"{instrument_name}_{best_verb['name']}"
                    frame_pairs.append(pair)
                    
                    # Visualization
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                    text_color = 'blue' if confidence >= CONFIDENCE_THRESHOLD else 'orange'
                    draw.text((x1, y1-25), 
                            f"{instrument_name}-{best_verb['name']}\n"
                            f"Conf: {confidence:.2f}, Verb: {best_verb['probability']:.2f}", 
                            fill=text_color, font=font)
            
            # Rest of the visualization code...
            return frame_pairs
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return []

    def evaluate(self):
        """
        Performs evaluation across all specified videos.
        Calculates metrics for instruments, verbs, and instrument-verb pairs.
        """
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
        
        for video in VIDEOS_TO_ANALYZE:
            print(f"\nProcessing {video}...")
            ground_truth = self.load_ground_truth(video)
            
            video_folder = os.path.join(self.dataset_dir, "videos", video)
            frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.png')])
            
            for frame_file in tqdm(frame_files, desc=f"Evaluating {video}"):
                frame_number = int(frame_file.split('.')[0])
                img_path = os.path.join(video_folder, frame_file)
                
                try:
                    frame_predictions, frame_metrics = self.evaluate_frame(
                        img_path,
                        ground_truth[frame_number],
                        save_visualization=True
                    )
                    
                    # Check if detections are present
                    has_predictions = any(bool(preds) for category_preds in frame_predictions.values() 
                                    for preds in category_preds.values())
                    
                    if not has_predictions:
                        # Add negative samples for ground truth annotations
                        for category in ['instruments', 'verbs', 'pairs']:
                            for item, count in ground_truth[frame_number][category].items():
                                if count > 0:
                                    metrics[category][item].append({
                                        'gt': True,
                                        'pred_confidence': 0.0
                                    })
                                    print(f"Missed {category}: {item} in frame {frame_number}")
                    else:
                        # Update metrics for found detections
                        for category in ['instruments', 'verbs', 'pairs']:
                            for item, predictions in frame_predictions[category].items():
                                if predictions:
                                    gt_count = ground_truth[frame_number][category][item]
                                    pred_confidence = max(predictions)
                                    
                                    metrics[category][item].append({
                                        'gt': gt_count > 0,
                                        'pred_confidence': pred_confidence
                                    })
                                    
                                    # Logging for False Positives
                                    if not gt_count > 0:
                                        print(f"False Positive {category}: {item} in frame {frame_number}")
                    
                    # Update total metrics
                    for category in ['instruments', 'verbs', 'pairs']:
                        total_metrics[category]['TP'] += frame_metrics[category]['TP']
                        total_metrics[category]['FP'] += frame_metrics[category]['FP']
                        total_metrics[category]['FN'] += frame_metrics[category]['FN']
                        
                except Exception as e:
                    print(f"Error processing frame {frame_number}: {str(e)}")
                    continue
                # Calculate final metrics
        results = {}
        print("\nFinal Evaluation Results:")
        print("========================")
        
        for category in ['instruments', 'verbs', 'pairs']:
            print(f"\n{category.upper()} METRICS:")
            print("-" * 20)
            
            # Global metrics
            category_total = total_metrics[category]
            print(f"Total True Positives: {category_total['TP']}")
            print(f"Total False Positives: {category_total['FP']}")
            print(f"Total False Negatives: {category_total['FN']}")
            
            if category_total['TP'] + category_total['FP'] > 0:
                precision = category_total['TP'] / (category_total['TP'] + category_total['FP'])
                recall = category_total['TP'] / (category_total['TP'] + category_total['FN'])
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"Overall Precision: {precision:.4f}")
                print(f"Overall Recall: {recall:.4f}")
                print(f"Overall F1-Score: {f1:.4f}")
            
            # Per-class metrics
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
                    
                    if len(predictions) > 0:
                        mean_conf = np.mean([p['pred_confidence'] for p in predictions])
                        print(f"Mean confidence: {mean_conf:.4f}")
            
            # Calculate mAP
            mean_ap = np.mean(list(category_aps.values())) if category_aps else 0
            results[category] = {
                'per_class_ap': category_aps,
                'mAP': mean_ap,
                'metrics': total_metrics[category]
            }
            print(f"\nmAP: {mean_ap:.4f}")
        
        # Save detailed results
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
        
def main():
    # Initialize ModelLoader
    try:
        loader = ModelLoader()
        
        # Load models
        yolo_model = loader.load_yolo_model()
        verb_model = loader.load_verb_model()
        
        # Dataset directory
        dataset_dir = str(loader.dataset_path)
        
        # Verify dataset structure
        labels_dir = os.path.join(dataset_dir, "labels")
        videos_dir = os.path.join(dataset_dir, "videos")
        
        print("\nDataset Structure Check:")
        print(f"Labels Directory: {labels_dir}")
        print(f"Videos Directory: {videos_dir}")
        
        # Verify directories exist
        if not os.path.exists(labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        if not os.path.exists(videos_dir):
            raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
        
        # List available videos
        available_videos = os.listdir(videos_dir)
        print("\nAvailable Videos:")
        for video in available_videos:
            print(f"- {video}")
        
        # Create Evaluator
        evaluator = HierarchicalEvaluator(
            yolo_model=yolo_model, 
            verb_model=verb_model, 
            dataset_dir=dataset_dir
        )
        
        # Run Evaluation
        results = evaluator.evaluate()
        
        print("\nEvaluation Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Error during initialization or evaluation: {str(e)}")

if __name__ == '__main__':
    main()