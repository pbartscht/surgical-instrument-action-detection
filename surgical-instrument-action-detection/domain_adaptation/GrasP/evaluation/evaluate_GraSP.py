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
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, accuracy_score
from collections import defaultdict

# Get the current script's directory and navigate to project root
current_dir = Path(__file__).resolve().parent  # GrasP/evaluation
hei_chole_dir = current_dir.parent  # GrasP
domain_adaptation_dir = hei_chole_dir.parent  # domain_adaptation
project_root = domain_adaptation_dir.parent  # surgical-instrument-action-detection
hierarchical_dir = project_root / "models" / "hierarchical-surgical-workflow"

# Add paths to Python path
sys.path.append(str(project_root))
sys.path.append(str(hierarchical_dir))

# Now try the imports
try:
    from verb_recognition.models.SurgicalActionNet import SurgicalVerbRecognition
    print("\n✓ Successfully imported SurgicalVerbRecognition")
except ImportError as e:
    print(f"\n✗ Failed to import SurgicalVerbRecognition: {str(e)}")

# Constants
CONFIDENCE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.3

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

# Mapping between CholecT50 and GraSP instruments
CHOLECT50_TO_GRASP_INSTRUMENT_MAPPING = {
    'grasper': ['Prograsp Forceps', 'Laparoscopic Grasper'],
    'bipolar': ['Bipolar Forceps'],
    'scissors': ['Monopolar Curved Scissors'],
    'clipper': ['Clip Applier'],
    'irrigator': ['Suction Instrument'],
    'hook': None  # No corresponding instrument in GraSP
}

# Reverse mapping for evaluation
GRASP_TO_CHOLECT50_INSTRUMENT_MAPPING = {
    'Prograsp Forceps': 'grasper',
    'Laparoscopic Grasper': 'grasper',
    'Bipolar Forceps': 'bipolar',
    'Monopolar Curved Scissors': 'scissors',
    'Clip Applier': 'clipper',
    'Suction Instrument': 'irrigator',
    'Large Needle Driver': None  # No corresponding instrument in CholecT50
}

CHOLECT50_TO_HEICHOLE_VERB_MAPPING = {
    'grasp': ['Hold', 'Still', 'Release'],
    'retract': ['Pull', 'Still'],
    'dissect': None,
    'coagulate': ['Cauterize'],
    'clip': ['Close'],
    'cut': ['Cut'],
    'aspirate': ['Suction'],
    'irrigate': ['Suction'],
    'pack': ['Push', 'Other'],
    'null_verb': ['Travel', 'Push', 'Open']
}

class ModelLoader:
    def __init__(self):
        self.project_root = project_root
        self.hierarchical_dir = hierarchical_dir
        self.setup_paths()

    def setup_paths(self):
        """Defines all important paths for the models"""
        # YOLO model path
        self.yolo_weights = self.hierarchical_dir / "Instrument-classification-detection" / "weights" / "instrument_detector" / "best_v35.pt"
        # Verb model path
        self.verb_model_path = self.hierarchical_dir / "verb_recognition/checkpoints/jumping-tree-47/last.ckpt"
        
        # Dataset path for GraSP
        self.dataset_path = Path("/data/Bartscht/GrasP/test")
        
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
        
class GraSPEvaluator:
    def __init__(self, yolo_model, verb_model, dataset_dir):
        """
        Initialize the GraSP evaluator.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = yolo_model
        self.verb_model = verb_model.to(self.device)
        self.verb_model.eval()
        self.dataset_dir = Path(dataset_dir)
        
        # Image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize data structures for GraSP dataset
        self.labels_dir = self.dataset_dir / "Labels"
        self.video_ids = sorted([f.stem for f in self.labels_dir.glob("*.json")])
        self.instrument_data = defaultdict(lambda: defaultdict(list))
        self.action_data = defaultdict(lambda: defaultdict(list))
        
        print(f"Initialized GraSP evaluator with dataset at: {self.dataset_dir}")
        print(f"Using device: {self.device}")

    def get_frame_annotations(self, video_id, frame_id):
        """
        Retrieves annotations for a specific frame with enhanced debugging.
        """
        print(f"\n--- Getting Annotations ---")
        print(f"Video: {video_id}, Frame: {frame_id}")
        print(f"Current instrument_data keys: {list(self.instrument_data.keys())}")

        # Ensure annotations are loaded for this video
        if video_id not in self.instrument_data or not self.instrument_data[video_id]:
            print(f"Loading annotations for {video_id}")
            self._load_video_annotations(video_id)
        
        # Convert frame_id to str
        frame_id = str(frame_id)
        
        # Debugging prints
        print(f"Instrument data for {video_id}: {dict(self.instrument_data[video_id])}")
        print(f"Action data for {video_id}: {dict(self.action_data[video_id])}")

        # Get instruments and actions for this frame
        instruments = self.instrument_data[video_id].get(frame_id, [])
        actions = self.action_data[video_id].get(frame_id, [])
        
        # Debug: Print results
        print(f"Found Instruments for frame {frame_id}: {instruments}")
        print(f"Found Actions for frame {frame_id}: {actions}")
        
        return instruments, actions

    def _load_video_annotations(self, video_id):
        json_file = self.labels_dir / f"{video_id}.json"
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Debugging: Print full JSON structure
            print(f"\n--- JSON Structure for {video_id} ---")
            print(json.dumps(list(data.keys()), indent=2))
            
            # More robust category parsing
            instrument_categories = {
                str(cat.get('id')): str(cat.get('name', 'Unknown')) 
                for cat in data.get('categories', {}).get('instruments', [])
            }
            action_categories = {
                str(cat.get('id')): str(cat.get('name', 'Unknown')) 
                for cat in data.get('categories', {}).get('actions', [])
            }
            
            # Extensive debugging prints
            print("\nInstrument Categories:")
            print(json.dumps(instrument_categories, indent=2))
            print("\nAction Categories:")
            print(json.dumps(action_categories, indent=2))
            
            # Reset data for this video
            self.instrument_data[video_id].clear()
            self.action_data[video_id].clear()
            
            # Debugging: Verify frames structure
            frames = data.get('frames', {})
            print(f"\nTotal frames in video {video_id}: {len(frames)}")
            print("Frame IDs:", list(frames.keys())[:10])  # Print first 10 frame IDs
            
            # More robust frame processing
            for frame_id, frame_data in frames.items():
                frame_id = str(frame_id)  # Ensure string
                
                # Debugging: Print frame details
                print(f"\nProcessing Frame {frame_id}")
                print("Frame Instruments:", frame_data.get('instruments', []))
                
                for instrument_ann in frame_data.get('instruments', []):
                    category_id = str(instrument_ann.get('category_id'))
                    
                    # Debugging: Instrument annotation details
                    print(f"Instrument Annotation: {instrument_ann}")
                    print(f"Category ID: {category_id}")
                    print(f"Available Categories: {list(instrument_categories.keys())}")
                    
                    if category_id in instrument_categories:
                        instr_name = instrument_categories[category_id]
                        self.instrument_data[video_id][frame_id].append(instr_name)
                        
                        # Process actions
                        for action_id in instrument_ann.get('actions', []):
                            action_id = str(action_id)
                            if action_id in action_categories:
                                action_name = action_categories[action_id]
                                self.action_data[video_id][frame_id].append({
                                    'instrument': instr_name,
                                    'action': action_name
                                })
            
            # Final debugging: Print loaded data
            print(f"\n--- Loaded Data for {video_id} ---")
            print("Instruments:", dict(self.instrument_data[video_id]))
            print("Actions:", dict(self.action_data[video_id]))
        
        except Exception as e:
            print(f"Detailed error in loading annotations for Video {video_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def evaluate_frame(self, img_path, frame_annotations, save_visualization=True):
            """
            Evaluates a single frame for GraSP dataset.
            
            Args:
                img_path: Path to the frame image
                frame_annotations: Ground truth annotations for this frame
                save_visualization: Whether to save visualization
            """
            frame_predictions = []
            frame_number = int(os.path.basename(img_path).split('.')[0])
            video_name = os.path.basename(os.path.dirname(img_path))
            
            img = Image.open(img_path)
            original_img = img.copy()
            draw = ImageDraw.Draw(original_img)
            
            try:
                # YOLO predictions
                yolo_results = self.yolo_model(img)
                valid_detections = []
                
                # Process YOLO detections
                for detection in yolo_results[0].boxes:
                    instrument_class = int(detection.cls)
                    confidence = float(detection.conf)
                    
                    if confidence >= CONFIDENCE_THRESHOLD:
                        # Get CholecT50 instrument name
                        cholect50_instrument = TOOL_MAPPING[instrument_class]
                        
                        # Map to GraSP instruments (could be multiple)
                        grasp_instruments = CHOLECT50_TO_GRASP_INSTRUMENT_MAPPING.get(cholect50_instrument)
                        
                        print(f"\n{'='*50}")
                        print(f"Frame {frame_number} Detection:")
                        print(f"CholecT50 Instrument: {cholect50_instrument}")
                        print(f"GraSP Instrument(s): {grasp_instruments}")
                        
                        if grasp_instruments:
                            if isinstance(grasp_instruments, str):
                                grasp_instruments = [grasp_instruments]
                            
                            for grasp_instrument in grasp_instruments:
                                valid_detections.append({
                                    'class': instrument_class,
                                    'confidence': confidence,
                                    'box': detection.xyxy[0],
                                    'name': grasp_instrument,
                                    'original_name': cholect50_instrument
                                })
                
                # Sort by confidence
                valid_detections.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Process each detection
                for detection in valid_detections:
                    grasp_instrument = detection['name']
                    cholect50_instrument = detection['original_name']
                    box = detection['box']
                    confidence = detection['confidence']
                    
                    # Get instrument crop for verb prediction
                    x1, y1, x2, y2 = map(int, box)
                    instrument_crop = img.crop((x1, y1, x2, y2))
                    crop_tensor = self.transform(instrument_crop).unsqueeze(0).to(self.device)
                    
                    # Predict verb using CholecT50 instrument name
                    verb_outputs = self.verb_model(crop_tensor, [cholect50_instrument])
                    verb_probs = verb_outputs['probabilities']
                    
                    # Get top verb predictions
                    for verb_idx in torch.topk(verb_probs[0], k=3).indices.cpu().numpy():
                        # Get CholecT50 verb and map to GraSP verb
                        cholect50_verb = VERB_MAPPING[VERB_MODEL_TO_EVAL_MAPPING[verb_idx]]
                        grasp_verbs = CHOLECT50_TO_HEICHOLE_VERB_MAPPING.get(cholect50_verb)
                        verb_confidence = float(verb_probs[0][verb_idx])
                        
                        print(f"\nVerb Prediction for {cholect50_instrument}:")
                        print(f"CholecT50 Verb: {cholect50_verb}")
                        print(f"GraSP Verb: {grasp_verbs}")
                        print(f"Confidence: {verb_confidence:.4f}")
                        
                        if grasp_verbs:
                            if isinstance(grasp_verbs, str):
                                grasp_verbs = [grasp_verbs]
                                
                            for grasp_verb in grasp_verbs:
                                prediction = {
                                    'frame_id': f"{video_name}_frame{frame_number}",
                                    'instrument': {
                                        'name': grasp_instrument,
                                        'confidence': confidence
                                    },
                                    'action': {
                                        'name': grasp_verb,
                                        'confidence': verb_confidence
                                    }
                                }
                                frame_predictions.append(prediction)
                                
                                # Visualization
                                if save_visualization:
                                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                                    text = f"{grasp_instrument}\n{grasp_verb}\nConf: {confidence:.2f}"
                                    draw.text((x1, y1-40), text, fill='blue')
                
                print(f"\n{'='*50}\n")
                
                # Save visualization
                if save_visualization:
                    viz_dir = os.path.join(self.dataset_dir, "visualizations")
                    os.makedirs(viz_dir, exist_ok=True)
                    save_path = os.path.join(viz_dir, f"{video_name}_frame{frame_number}.jpg")
                    original_img.save(save_path)
                
                return frame_predictions
                
            except Exception as e:
                print(f"Error processing frame {frame_number}: {str(e)}")
                return []
def analyze_grasp_distribution(dataset_path):
    """
    Analyzes the distribution of instruments and actions in GraSP training dataset.
    
    Args:
        dataset_path: Path to the GraSP dataset directory
    """
    labels_dir = Path(dataset_path) / "Labels"
    video_ids = [f.stem for f in labels_dir.glob("*.json")]
    video_ids.sort()

    # Initialize counters
    total_frames = 0
    instrument_counts = defaultdict(int)
    action_counts = defaultdict(int)
    instrument_action_pairs = defaultdict(lambda: defaultdict(int))

    # Process each video
    for video_id in video_ids:
        json_file = labels_dir / f"{video_id}.json"
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Get categories lookup
                instrument_categories = {
                    cat['id']: cat['name'] 
                    for cat in data['categories']['instruments']
                }
                action_categories = {
                    cat['id']: cat['name'] 
                    for cat in data['categories']['actions']
                }
                
                frames = data.get('frames', {})
                total_frames += len(frames)
                
                # Process each frame
                for frame_data in frames.values():
                    for instrument_ann in frame_data.get('instruments', []):
                        category_id = instrument_ann.get('category_id')
                        if category_id is not None:
                            instr_name = instrument_categories[category_id]
                            instrument_counts[instr_name] += 1
                            
                            # Process actions for this instrument
                            for action_id in instrument_ann.get('actions', []):
                                action_name = action_categories[action_id]
                                action_counts[action_name] += 1
                                instrument_action_pairs[instr_name][action_name] += 1
        
        except Exception as e:
            print(f"Error processing {video_id}: {str(e)}")

    # Print results
    print(f"\nTotal Frames Analyzed: {total_frames}")
    
    # Print instrument frequencies
    print("\nINSTRUMENT FREQUENCIES:")
    for instr_name, count in sorted(instrument_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_frames) * 100
        print(f"{instr_name:30s} {count:8d} {percentage:11.2f}%")
        
        # Print associated actions
        if instrument_action_pairs[instr_name]:
            print("\n  Associated Actions:")
            for action, action_count in sorted(instrument_action_pairs[instr_name].items(), key=lambda x: x[1], reverse=True):
                action_percentage = (action_count / count) * 100
                print(f"    {action:26s}: {action_count:6d} ({action_percentage:5.2f}%)")
            print()

    # Print overall action frequencies
    print("\nOVERALL ACTION FREQUENCIES:")
    for action_name, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_frames) * 100
        print(f"{action_name:30s} {count:8d} {percentage:11.2f}%")


class MetricsCalculator:
    def __init__(self):
        self.reset_metrics()
        
    def reset_metrics(self):
        """Reset all metrics counters"""
        self.instrument_metrics = {
            'true_positives': defaultdict(int),
            'false_positives': defaultdict(int),
            'false_negatives': defaultdict(int)
        }
        self.action_metrics = {
            'true_positives': defaultdict(int),
            'false_positives': defaultdict(int),
            'false_negatives': defaultdict(int)
        }
        self.instrument_predictions = []
        self.instrument_ground_truth = []
        self.action_predictions = []
        self.action_ground_truth = []
        self.total_frames = 0
        self.gt_instrument_counts = defaultdict(int)  # Neu: Zählt Ground Truth Instrumente
        self.gt_action_counts = defaultdict(int)      # Neu: Zählt Ground Truth Aktionen
        self.pred_instrument_counts = defaultdict(int)
        self.pred_action_counts = defaultdict(int)

    def update_metrics(self, predictions, ground_truth):
        """Update metrics for each frame"""
        self.total_frames += 1
        gt_instruments, gt_actions = ground_truth
        
        # Zähle Ground Truth
        for instr in gt_instruments:
            self.gt_instrument_counts[instr] += 1
            
        for action in gt_actions:
            action_key = f"{action['instrument']}_{action['action']}"
            self.gt_action_counts[action_key] += 1
        
        # Process instrument predictions
        predicted_instruments = set()
        for pred in predictions:
            instr_name = pred['instrument']['name']
            predicted_instruments.add(instr_name)
            self.pred_instrument_counts[instr_name] += 1
            
            # Add prediction confidence for mAP calculation
            self.instrument_predictions.append({
                'label': instr_name,
                'confidence': pred['instrument']['confidence']
            })
            
            # Process action prediction
            action_name = pred['action']['name']
            action_key = f"{instr_name}_{action_name}"
            self.pred_action_counts[action_key] += 1
            
            self.action_predictions.append({
                'label': action_key,
                'confidence': pred['action']['confidence']
            })

        # Update metrics
        self._update_instrument_metrics(predicted_instruments, set(gt_instruments))
        self._update_action_metrics(predictions, gt_actions)

    def _update_instrument_metrics(self, predicted_instruments, gt_instruments):
        """Update metrics specifically for instruments"""
        for instr in predicted_instruments:
            if instr in gt_instruments:
                self.instrument_metrics['true_positives'][instr] += 1
            else:
                self.instrument_metrics['false_positives'][instr] += 1
        
        for instr in gt_instruments:
            self.instrument_ground_truth.append(instr)
            if instr not in predicted_instruments:
                self.instrument_metrics['false_negatives'][instr] += 1

    def _update_action_metrics(self, predictions, gt_actions):
        """Update metrics specifically for actions"""
        predicted_actions = {(p['instrument']['name'], p['action']['name']): p['action']['confidence'] 
                           for p in predictions}
        gt_action_tuples = {(a['instrument'], a['action']) for a in gt_actions}
        
        for (instr, action), conf in predicted_actions.items():
            action_tuple = (instr, action)
            if action_tuple in gt_action_tuples:
                self.action_metrics['true_positives'][action_tuple] += 1
            else:
                self.action_metrics['false_positives'][action_tuple] += 1
        
        for action_tuple in gt_action_tuples:
            self.action_ground_truth.append(action_tuple)
            if action_tuple not in predicted_actions:
                self.action_metrics['false_negatives'][action_tuple] += 1

    def calculate_metrics(self):
        """Calculate detailed metrics"""
        metrics = {
            'instruments': self._calculate_category_metrics(
                'instruments',
                self.instrument_metrics,
                self.instrument_predictions,
                self.instrument_ground_truth,
                self.gt_instrument_counts,
                self.pred_instrument_counts
            ),
            'actions': self._calculate_category_metrics(
                'actions',
                self.action_metrics,
                self.action_predictions,
                self.action_ground_truth,
                self.gt_action_counts,
                self.pred_action_counts
            )
        }
        return metrics

    def _calculate_category_metrics(self, category_type, category_metrics, predictions, 
                                  ground_truth, gt_counts, pred_counts):
        """Calculate metrics for a specific category"""
        per_class_metrics = {}
        
        # Combine all classes from both GT and predictions
        all_classes = set(gt_counts.keys()) | set(pred_counts.keys())
        
        for cls in all_classes:
            tp = category_metrics['true_positives'][cls]
            fp = category_metrics['false_positives'][cls]
            fn = category_metrics['false_negatives'][cls]
            
            gt_count = gt_counts[cls]
            pred_count = pred_counts[cls]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate AP
            cls_predictions = [p for p in predictions if p['label'] == cls]
            cls_ground_truth = [1 if g == cls else 0 for g in ground_truth]
            
            if cls_predictions:
                confidences = [p['confidence'] for p in cls_predictions]
                y_true = np.array(cls_ground_truth[:len(confidences)])
                y_score = np.array(confidences)
                ap = average_precision_score(y_true, y_score) if len(y_true) > 0 else 0
            else:
                ap = 0
            
            per_class_metrics[cls] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'ap': ap,
                'support': gt_count,
                'predictions': pred_count
            }

        # Calculate macro averages for classes with ground truth
        classes_with_gt = [cls for cls in all_classes if gt_counts[cls] > 0]
        if classes_with_gt:
            macro_metrics = {
                'precision': np.mean([per_class_metrics[cls]['precision'] for cls in classes_with_gt]),
                'recall': np.mean([per_class_metrics[cls]['recall'] for cls in classes_with_gt]),
                'f1': np.mean([per_class_metrics[cls]['f1'] for cls in classes_with_gt]),
                'mAP': np.mean([per_class_metrics[cls]['ap'] for cls in classes_with_gt])
            }
        else:
            macro_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'mAP': 0}

        return {'per_class': per_class_metrics, 'macro': macro_metrics}
    
def print_evaluation_report(metrics, total_frames):
    """Print a detailed evaluation report"""
    print("\n" + "="*90)
    print("GRASP EVALUATION REPORT")
    print("="*90)
    
    # Print Ground Truth Distribution
    print("\nGROUND TRUTH DISTRIBUTION:")
    print("-"*70)
    print(f"{'Category':20s} {'Count':>10s} {'Percentage':>12s}")
    print("-"*70)
    
    # Print metrics for each category
    for category in ['instruments', 'actions']:
        print(f"\n{category.upper()}:")
        metrics_data = metrics[category]['per_class']
        
        for cls, cls_metrics in sorted(metrics_data.items()):
            count = cls_metrics['support']
            percentage = (count / total_frames) * 100
            name = cls if isinstance(cls, str) else f"{cls[0]}_{cls[1]}"
            print(f"{name:20s} {count:10d} {percentage:11.2f}%")
    
    # Print Model Performance
    print("\n" + "="*90)
    print("MODEL PERFORMANCE:")
    print("="*90)
    
    header = f"{'Category':20s} {'Predictions':>12s} {'AP Score':>10s} {'F1 Score':>10s} {'Precision':>10s} {'Recall':>10s}"
    print(header)
    print("-"*90)
    
    for category in ['instruments', 'actions']:
        print(f"\n{category.upper()}:")
        metrics_data = metrics[category]['per_class']
        
        for cls, cls_metrics in sorted(metrics_data.items()):
            name = cls if isinstance(cls, str) else f"{cls[0]}_{cls[1]}"
            pred_count = cls_metrics['predictions']
            percentage = (pred_count / total_frames) * 100
            
            print(f"{name:20s} {pred_count:10d} ({percentage:6.2f}%) "
                  f"{cls_metrics['ap']:8.4f} {cls_metrics['f1']:8.4f} "
                  f"{cls_metrics['precision']:8.4f} {cls_metrics['recall']:8.4f}")
    
    # Print Mean Scores
    print("\nMEAN SCORES:")
    print("="*50)
    for category in ['instruments', 'actions']:
        print(f"\n{category.upper()}:")
        macro = metrics[category]['macro']
        print(f"mAP: {macro['mAP']:.4f}")
        print(f"F1:  {macro['f1']:.4f}")
        print(f"Precision: {macro['precision']:.4f}")
        print(f"Recall: {macro['recall']:.4f}")

def comprehensive_video_evaluation(dataset_dir, specific_video_id=None):
    """
    Comprehensive model evaluation for a single video
    
    Args:
        dataset_dir: Path to the GraSP dataset directory
        specific_video_id: Optional specific video ID to analyze
    """
    # Initialize logging and model loading
    print("\n" + "="*50)
    print("COMPREHENSIVE VIDEO MODEL EVALUATION")
    print("="*50)

    # Initialize ModelLoader
    loader = ModelLoader()
    
    # Load models
    print("\n--- Loading Models ---")
    yolo_model = loader.load_yolo_model()
    verb_model = loader.load_verb_model()
    
    # Select video
    labels_dir = Path(dataset_dir) / "Labels"
    
    if not specific_video_id:
        video_ids = [f.stem for f in labels_dir.glob("*.json")]
        if not video_ids:
            raise ValueError(f"No JSON files found in {labels_dir}")
        specific_video_id = video_ids[0]
    
    print(f"\nSelected Video: {specific_video_id}")
    
    # Initialize Evaluator
    evaluator = GraSPEvaluator(yolo_model, verb_model, dataset_dir)
    metrics_calculator = MetricsCalculator()
    
    # Prepare paths
    video_folder = Path(dataset_dir) / "Videos" / specific_video_id
    frame_files = sorted(list(video_folder.glob("*.jpg")))
    
    # Detailed tracking
    total_frames = 0
    ground_truth_summary = {
        'total_frames': 0,
        'instruments': defaultdict(int),
        'actions': defaultdict(int)
    }
    model_predictions_summary = {
        'total_frames': 0,
        'instruments': defaultdict(int),
        'actions': defaultdict(int)
    }
    
    # Process each frame
    for frame_file in tqdm(frame_files, desc=f"Processing {specific_video_id}"):
        total_frames += 1
        frame_number = int(frame_file.stem)
        
        # Get Ground Truth annotations
        try:
            frame_gt = evaluator.get_frame_annotations(specific_video_id, frame_number)
            
            # Update ground truth summary
            ground_truth_summary['total_frames'] += 1
            for instrument in frame_gt[0]:
                ground_truth_summary['instruments'][instrument] += 1
            for action in frame_gt[1]:
                ground_truth_summary['actions'][f"{action['instrument']}_{action['action']}"] += 1
            
            # Evaluate frame
            frame_predictions = evaluator.evaluate_frame(
                str(frame_file), 
                frame_gt, 
                save_visualization=False
            )
            
            # Update model predictions summary
            if frame_predictions:
                model_predictions_summary['total_frames'] += 1
                for pred in frame_predictions:
                    instrument = pred['instrument']['name']
                    action = pred['action']['name']
                    
                    model_predictions_summary['instruments'][instrument] += 1
                    model_predictions_summary['actions'][f"{instrument}_{action}"] += 1
                
                # Update metrics
                metrics_calculator.update_metrics(frame_predictions, frame_gt)
        
        except Exception as e:
            print(f"Error processing frame {frame_number}: {str(e)}")
    
    # Calculate final metrics
    final_metrics = metrics_calculator.calculate_metrics()
    
    # Detailed Comparison Report
    print("\n" + "="*70)
    print("DETAILED MODEL EVALUATION REPORT")
    print("="*70)
    
    # Instrument Comparison
    print("\nINSTRUMENT COMPARISON:")
    print(f"{'Instrument':30s} {'Ground Truth':>15s} {'Model Prediction':>20s}")
    print("-"*70)
    
    all_instruments = set(list(ground_truth_summary['instruments'].keys()) + 
                          list(model_predictions_summary['instruments'].keys()))
    
    for instrument in sorted(all_instruments):
        gt_count = ground_truth_summary['instruments'].get(instrument, 0)
        pred_count = model_predictions_summary['instruments'].get(instrument, 0)
        
        print(f"{instrument:30s} {gt_count:15d} {pred_count:20d}")
    
    # Actions Comparison
    print("\nACTION COMPARISON:")
    print(f"{'Action':50s} {'Ground Truth':>15s} {'Model Prediction':>20s}")
    print("-"*90)
    
    all_actions = set(list(ground_truth_summary['actions'].keys()) + 
                      list(model_predictions_summary['actions'].keys()))
    
    for action in sorted(all_actions):
        gt_count = ground_truth_summary['actions'].get(action, 0)
        pred_count = model_predictions_summary['actions'].get(action, 0)
        
        print(f"{action:50s} {gt_count:15d} {pred_count:20d}")
    
    # Print Evaluation Metrics
    print("\n" + "="*70)
    print("MODEL PERFORMANCE METRICS")
    print("="*70)
    print_evaluation_report(final_metrics, total_frames)
    
    return ground_truth_summary, model_predictions_summary, final_metrics

def debug_ground_truth_loading(dataset_dir, video_id):
    """
    Comprehensive debugging of ground truth data loading
    
    Args:
        dataset_dir: Path to the GraSP dataset directory
        video_id: Specific video ID to debug
    """
    labels_dir = Path(dataset_dir) / "Labels"
    json_file = labels_dir / f"{video_id}.json"
    
    print(f"\n{'='*50}")
    print(f"DEBUGGING GROUND TRUTH FOR VIDEO: {video_id}")
    print(f"JSON File Path: {json_file}")
    print(f"{'='*50}")
    
    # Check if file exists
    if not json_file.exists():
        print(f"❌ ERROR: JSON file does not exist at {json_file}")
        return
    
    try:
        # Read and parse JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Print full JSON structure
        print("\n--- FULL JSON STRUCTURE ---")
        print(json.dumps(list(data.keys()), indent=2))
        
        # Detailed category parsing
        print("\n--- CATEGORIES ---")
        categories = data.get('categories', {})
        
        # Print instruments categories
        print("\nInstrument Categories:")
        instrument_categories = categories.get('instruments', [])
        for cat in instrument_categories:
            print(f"ID: {cat.get('id')}, Name: {cat.get('name')}")
        
        # Print action categories
        print("\nAction Categories:")
        action_categories = categories.get('actions', [])
        for cat in action_categories:
            print(f"ID: {cat.get('id')}, Name: {cat.get('name')}")
        
        # Check frames
        frames = data.get('frames', {})
        print(f"\nTotal Frames: {len(frames)}")
        
        # Sample first few frames
        print("\n--- SAMPLE FRAME DETAILS ---")
        sample_frames = list(frames.keys())[:5]
        for frame_id in sample_frames:
            frame_data = frames[frame_id]
            print(f"\nFrame {frame_id}:")
            print("Instruments:", frame_data.get('instruments', []))
            print("Actions:", [instr.get('actions', []) for instr in frame_data.get('instruments', [])])
        
    except Exception as e:
        print(f"❌ ERROR during JSON parsing: {str(e)}")
        import traceback
        traceback.print_exc()

def comprehensive_model_evaluation(dataset_dir, video_id):
    """
    Comprehensive model evaluation for a single video
    
    Args:
        dataset_dir: Path to the GraSP dataset directory
        video_id: Specific video ID to evaluate
    """
    # Initialize logging and model loading
    print("\n" + "="*50)
    print(f"MODEL EVALUATION FOR VIDEO: {video_id}")
    print("="*50)

    # Initialize ModelLoader
    loader = ModelLoader()
    
    # Load models
    print("\n--- Loading Models ---")
    yolo_model = loader.load_yolo_model()
    verb_model = loader.load_verb_model()
    
    # Prepare paths
    video_folder = Path(dataset_dir) / "Videos" / video_id
    frame_files = sorted(list(video_folder.glob("*.jpg")))
    
    # Initialize Evaluator
    evaluator = GraSPEvaluator(yolo_model, verb_model, dataset_dir)
    
    # Ground Truth Categories
    instrument_categories = {
        1: 'Bipolar Forceps',
        2: 'Prograsp Forceps',
        3: 'Large Needle Driver',
        4: 'Monopolar Curved Scissors', 
        5: 'Suction Instrument',
        6: 'Clip Applier',
        7: 'Laparoscopic Grasper'
    }
    
    action_categories = {
        1: 'Cauterize',
        2: 'Close',
        3: 'Cut',
        4: 'Grasp',
        5: 'Hold',
        6: 'Open',
        7: 'Open Something',
        8: 'Pull',
        9: 'Push',
        10: 'Release',
        11: 'Still',
        12: 'Suction',
        13: 'Travel',
        14: 'Other'
    }
    
    # Tracking metrics
    total_frames = 0
    model_predictions = {
        'instruments': defaultdict(int),
        'actions': defaultdict(int)
    }
    ground_truth = {
        'instruments': defaultdict(int),
        'actions': defaultdict(int)
    }
    
    # Process each frame
    for frame_file in tqdm(frame_files, desc=f"Evaluating {video_id}"):
        total_frames += 1
        frame_number = int(frame_file.stem)
        
        try:
            # Get Ground Truth annotations
            frame_gt = evaluator.get_frame_annotations(video_id, frame_number)
            
            # Process ground truth
            for instrument_ann in frame_gt[0]:
                ground_truth['instruments'][instrument_ann] += 1
            
            for action in frame_gt[1]:
                ground_truth['actions'][f"{action['instrument']}_{action['action']}"] += 1
            
            # Evaluate frame
            frame_predictions = evaluator.evaluate_frame(
                str(frame_file), 
                frame_gt, 
                save_visualization=False
            )
            
            # Process model predictions
            for pred in frame_predictions:
                instrument = pred['instrument']['name']
                action = pred['action']['name']
                
                model_predictions['instruments'][instrument] += 1
                model_predictions['actions'][f"{instrument}_{action}"] += 1
        
        except Exception as e:
            print(f"Error processing frame {frame_number}: {str(e)}")
    
    # Print Detailed Comparison
    print("\n" + "="*70)
    print("GROUND TRUTH VS MODEL PREDICTIONS")
    print("="*70)
    
    # Instrument Comparison
    print("\nINSTRUMENT COMPARISON:")
    print(f"{'Instrument':30s} {'Ground Truth':>15s} {'Model Prediction':>20s}")
    print("-"*70)
    
    all_instruments = set(list(ground_truth['instruments'].keys()) + 
                          list(model_predictions['instruments'].keys()))
    
    for instrument in sorted(all_instruments):
        gt_count = ground_truth['instruments'].get(instrument, 0)
        pred_count = model_predictions['instruments'].get(instrument, 0)
        gt_percentage = (gt_count / total_frames) * 100
        pred_percentage = (pred_count / total_frames) * 100
        
        print(f"{instrument:30s} {gt_count:10d} ({gt_percentage:5.2f}%) {pred_count:15d} ({pred_percentage:5.2f}%)")
    
    # Action Comparison
    print("\nACTION COMPARISON:")
    print(f"{'Action':50s} {'Ground Truth':>15s} {'Model Prediction':>20s}")
    print("-"*90)
    
    all_actions = set(list(ground_truth['actions'].keys()) + 
                      list(model_predictions['actions'].keys()))
    
    for action in sorted(all_actions):
        gt_count = ground_truth['actions'].get(action, 0)
        pred_count = model_predictions['actions'].get(action, 0)
        gt_percentage = (gt_count / total_frames) * 100
        pred_percentage = (pred_count / total_frames) * 100
        
        print(f"{action:50s} {gt_count:10d} ({gt_percentage:5.2f}%) {pred_count:15d} ({pred_percentage:5.2f}%)")
    
    return ground_truth, model_predictions, total_frames

def main():
    """Main function to run comprehensive model evaluation"""
    try:
        # Initialize ModelLoader to get dataset path
        loader = ModelLoader()
        dataset_dir = str(loader.dataset_path)
        
        # Get first video
        labels_dir = Path(dataset_dir) / "Labels"
        video_ids = [f.stem for f in labels_dir.glob("*.json")]
        
        if not video_ids:
            raise ValueError(f"No JSON files found in {labels_dir}")
        
        # Evaluate first video
        first_video = video_ids[0]
        comprehensive_model_evaluation(dataset_dir, first_video)
        
    except Exception as main_error:
        print(f"❌ Critical Error during model evaluation: {str(main_error)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()