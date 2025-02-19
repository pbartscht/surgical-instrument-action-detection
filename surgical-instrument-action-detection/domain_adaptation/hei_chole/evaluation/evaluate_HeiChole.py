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
current_dir = Path(__file__).resolve().parent  # hei_chole/evaluation
hei_chole_dir = current_dir.parent  # hei_chole
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
CONFIDENCE_THRESHOLD = 0.0001
IOU_THRESHOLD = 0.3

# Global mappings
TOOL_MAPPING = {
    0: 'grasper', 1: 'bipolar', 2: 'hook', 
    3: 'scissors', 4: 'clipper', 5: 'irrigator'
}

IGNORED_INSTRUMENTS = {
    6: 'specimen_bag'  # Index: Name der zu ignorierenden Instrumente
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

# Constants for Dataset Mappings
CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING = {
    'grasper': 'grasper',
    'bipolar': 'coagulation',
    'clipper': 'clipper',
    'hook': 'coagulation',  # revised: hook is also a coagulation instrument
    'scissors': 'scissors',
    'irrigator': 'suction_irrigation'
}

CHOLECT50_TO_HEICHOLE_VERB_MAPPING = {
    'grasp': 'grasp',
    'retract': 'hold',
    'dissect': 'hold',
    'coagulate': 'hold',
    'clip': 'clip',
    'cut': 'cut',
    'irrigate': 'suction_irrigation',
    'aspirate': 'hold',
    'pack': 'hold',
    'null_verb': 'hold'
}

HEICHOLE_SPECIFIC_INSTRUMENTS = {
    'specimen_bag',
    'stapler'
}



class ModelLoader:
    def __init__(self):
        self.project_root = project_root
        self.hierarchical_dir = hierarchical_dir
        self.setup_paths()

    def setup_paths(self):
        """Defines all important paths for the models"""
        # YOLO model path
        #self.yolo_weights = self.hierarchical_dir / "Instrument-classification-detection" / "weights" / "instrument_detector" / "best_v35.pt"
        self.yolo_weights = Path("/home/Bartscht/YOLO/surgical-instrument-action-detection/domain_adaptation/hei_chole/experiments/finalfinal/heichole_transfer_balanced_instruments/transfer_learning/weights/epoch30.pt")
        # Verb model path
        self.verb_model_path = self.hierarchical_dir / "verb_recognition/checkpoints/jumping-tree-47/last.ckpt"
        
        # Dataset path for HeiChole, wichtig hier ändern!! 
        self.dataset_path = Path("/data/Bartscht/HeiChole/domain_adaptation/test")
        
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


class HeiCholeEvaluator:
    def __init__(self, yolo_model, verb_model, dataset_dir):
        """
        Initialize the HeiChole evaluator.
        
        Args:
            yolo_model: Pre-trained YOLO model for instrument detection
            verb_model: Pre-trained verb recognition model
            dataset_dir: Path to HeiChole dataset
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = yolo_model
        self.verb_model = verb_model.to(self.device)
        self.verb_model.eval()
        self.dataset_dir = dataset_dir
        
        # Image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def map_cholect50_prediction(self, instrument, verb):
        """
        Maps CholecT50 predictions to HeiChole format.
        """
        mapped_instrument = CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING.get(instrument)
        mapped_verb = CHOLECT50_TO_HEICHOLE_VERB_MAPPING.get(verb)
        return mapped_instrument, mapped_verb

    def load_ground_truth(self, video):
        """
        Loads binary ground truth annotations for HeiChole dataset.
        """
        labels_folder = os.path.join(self.dataset_dir, "Labels")
        json_file = os.path.join(labels_folder, f"{video}.json")
        
        frame_annotations = defaultdict(lambda: {
            'instruments': defaultdict(int),
            'actions': defaultdict(int)
        })
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                frames = data.get('frames', {})
                
                for frame_num, frame_data in frames.items():
                    frame_number = int(frame_num)
                    
                    # Process instruments (binary)
                    instruments = frame_data.get('instruments', {})
                    for instr_name, present in instruments.items():
                        frame_annotations[frame_number]['instruments'][instr_name] = 1 if present > 0 else 0
                    
                    # Process actions (binary)
                    actions = frame_data.get('actions', {})
                    for action_name, present in actions.items():
                        frame_annotations[frame_number]['actions'][action_name] = 1 if present > 0 else 0
                
                return frame_annotations
                
        except Exception as e:
            print(f"Error loading annotations: {str(e)}")
            raise

    def evaluate_frame(self, img_path, ground_truth, save_visualization=True):
        """
        Evaluates a single frame and maps predictions to HeiChole format.
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
                    if instrument_class in IGNORED_INSTRUMENTS:
                        print(f"\n{'='*50}")
                        print(f"Frame {frame_number}: Skipping ignored instrument {IGNORED_INSTRUMENTS[instrument_class]}")
                        continue
                    
                    try:
                        cholect50_instrument = TOOL_MAPPING[instrument_class]
                    except KeyError:
                        print(f"\nWarning: Unknown instrument class {instrument_class}, skipping...")
                        continue
                    
                    mapped_instrument = CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING.get(cholect50_instrument)
                    
                    if mapped_instrument:
                        valid_detections.append({
                            'class': instrument_class,
                            'confidence': confidence,
                            'box': detection.xyxy[0],
                            'name': mapped_instrument,
                            'original_name': cholect50_instrument
                        })
            
            # Sort by confidence
            valid_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Process each detection
            for detection in valid_detections:
                mapped_instrument = detection['name']
                original_instrument = detection['original_name']
                box = detection['box']
                confidence = detection['confidence']
                
                # Get instrument crop for verb prediction
                x1, y1, x2, y2 = map(int, box)
                instrument_crop = img.crop((x1, y1, x2, y2))
                crop_tensor = self.transform(instrument_crop).unsqueeze(0).to(self.device)
                
                # Predict verb using original CholecT50 instrument name
                verb_outputs = self.verb_model(crop_tensor, [original_instrument])
                verb_probs = verb_outputs['probabilities']
                
                # Get top verb predictions
                top_verbs = []
                for verb_idx in torch.topk(verb_probs[0], k=3).indices.cpu().numpy():
                    cholect50_verb = VERB_MAPPING[VERB_MODEL_TO_EVAL_MAPPING[verb_idx]]
                    mapped_verb = CHOLECT50_TO_HEICHOLE_VERB_MAPPING.get(cholect50_verb)
                    verb_confidence = float(verb_probs[0][verb_idx])
                    
                    if mapped_verb is not None:
                        top_verbs.append({
                            'name': mapped_verb,
                            'probability': verb_confidence
                        })
                
                # Create prediction for best verb
                if top_verbs:
                    best_verb = max(top_verbs, key=lambda x: x['probability'])
                    verb_name = best_verb['name']
                    verb_conf = best_verb['probability']
                    
                    prediction = {
                        'frame_id': f"{video_name}_frame{frame_number}",
                        'instrument': {
                            'name': mapped_instrument,
                            'confidence': confidence,
                            'binary_pred': 1 if confidence >= CONFIDENCE_THRESHOLD else 0
                        },
                        'action': {
                            'name': verb_name,
                            'confidence': verb_conf,
                            'binary_pred': 1 if verb_conf >= CONFIDENCE_THRESHOLD else 0
                        }
                    }
                    frame_predictions.append(prediction)
                    
                    # Visualization
                    if save_visualization:
                        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                        text = f"{mapped_instrument}\n{verb_name}\nConf: {confidence:.2f}"
                        draw.text((x1, y1-40), text, fill='blue')
            
            # Save visualization
            if save_visualization:
                viz_dir = os.path.join(self.dataset_dir, "visualizations")
                os.makedirs(viz_dir, exist_ok=True)
                save_path = os.path.join(viz_dir, f"{video_name}_frame{frame_number}.png")
                original_img.save(save_path)
            
            return frame_predictions
            
        except Exception as e:
            print(f"Error processing frame {frame_number}: {str(e)}")
            return []
        
class BinaryMetricsCalculator:
    def __init__(self, confidence_threshold=0.0001):
        self.confidence_threshold = confidence_threshold
        
        # Define fixed order of labels matching ground truth JSON structure
        self.instrument_labels = [
            'grasper',
            'clipper', 
            'coagulation',
            'scissors',
            'suction_irrigation',
            'specimen_bag',
            'stapler'
        ]
        
        self.action_labels = [
            'grasp',
            'hold',
            'cut',
            'clip'
        ]

    def calculate_metrics(self, predictions_per_frame, ground_truth):
        results = {
            'instruments': {'per_class': {}, 'mean_metrics': {}},
            'actions': {'per_class': {}, 'mean_metrics': {}}
        }
        
        all_frame_numbers = sorted(list(ground_truth.keys()))
        num_frames = len(all_frame_numbers)
        frame_to_idx = {frame: idx for idx, frame in enumerate(all_frame_numbers)}
        
        for category, label_list in [('instruments', self.instrument_labels), 
                                   ('actions', self.action_labels)]:
            # Three matrices: ground truth, binary predictions, and confidence scores
            y_true = np.zeros((num_frames, len(label_list)), dtype=np.int32)
            y_pred = np.zeros((num_frames, len(label_list)), dtype=np.int32)
            y_scores = np.zeros((num_frames, len(label_list)), dtype=np.float32)  # Neu!
            
            # Fill ground truth matrix
            for frame_num, frame_data in ground_truth.items():
                frame_idx = frame_to_idx[frame_num]
                for label_idx, label in enumerate(label_list):
                    if label in frame_data[category]:
                        y_true[frame_idx, label_idx] = frame_data[category][label]
            
            # Fill prediction and scores matrices
            for frame_id, preds in predictions_per_frame.items():
                frame_num = int(frame_id.split('_frame')[1])
                if frame_num in frame_to_idx:
                    frame_idx = frame_to_idx[frame_num]
                    for pred in preds:
                        if category == 'instruments':
                            name = pred['instrument']['name']
                            conf = pred['instrument']['confidence']
                        else:
                            name = pred['action']['name']
                            conf = pred['action']['confidence']
                            
                        try:
                            label_idx = label_list.index(name)
                            y_scores[frame_idx, label_idx] = conf  # Speichere confidence score
                            y_pred[frame_idx, label_idx] = 1 if conf >= self.confidence_threshold else 0
                        except ValueError:
                            continue
            
            # Calculate metrics for each class
            f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
            precision_scores = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_scores = recall_score(y_true, y_pred, average=None, zero_division=0)
            
            # Count instances per class
            ins_count_pred = np.sum(y_pred, axis=0)
            ins_count_gt = np.sum(y_true, axis=0)
            
            # Calculate per-class metrics
            overall_f1 = 0
            overall_ap = 0
            class_count = 0
            
            for i, label in enumerate(label_list):
                # Wichtig: AP wird mit confidence scores berechnet!
                ap = average_precision_score(y_true[:, i], y_scores[:, i])
                
                results[category]['per_class'][label] = {
                    'f1_score': float(f1_scores[i]),
                    'precision': float(precision_scores[i]),
                    'recall': float(recall_scores[i]),
                    'ap_score': float(ap),
                    'support': int(ins_count_gt[i]),
                    'predictions': int(ins_count_pred[i])
                }
                
                if ins_count_gt[i] > 0:  # Nur Klassen mit Ground Truth
                    overall_f1 += f1_scores[i]
                    overall_ap += ap
                    class_count += 1
            
            # Calculate mean metrics
            if class_count > 0:
                results[category]['mean_metrics'] = {
                    'mean_f1': overall_f1 / class_count,
                    'mean_precision': np.mean(precision_scores[precision_scores > 0]),
                    'mean_recall': np.mean(recall_scores[recall_scores > 0]),
                    'mean_ap': overall_ap / class_count
                }
        
        return results
def analyze_label_distribution(dataset_dir, videos):
    """
    Analyzes the distribution of ground truth labels, including zero occurrences.
    """
    # Define all possible classes (from mapping)
    all_possible_instruments = {
        'grasper', 'coagulation', 'clipper', 'scissors', 
        'suction_irrigation', 'specimen_bag', 'stapler'
    }
    
    all_possible_actions = {
        'grasp', 'hold', 'clip', 'cut'
    }
    
    # Initialize counters
    frequencies = {
        'instruments': defaultdict(int),
        'actions': defaultdict(int)
    }
    
    total_frames = 0
    
    print("\nAnalyzing ground truth label distribution...")
    
    # Process each video
    for video in videos:
        print(f"\nProcessing {video}...")
        json_file = os.path.join(dataset_dir, "Labels", f"{video}.json")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                frames = data.get('frames', {})
                total_frames += len(frames)
                
                # Count occurrences
                for frame_data in frames.values():
                    # Process instruments
                    instruments = frame_data.get('instruments', {})
                    for instr, present in instruments.items():
                        if present > 0:
                            frequencies['instruments'][instr] += 1
                    
                    # Process actions
                    actions = frame_data.get('actions', {})
                    for action, present in actions.items():
                        if present > 0:
                            frequencies['actions'][action] += 1
        
        except Exception as e:
            print(f"Error processing {video}: {str(e)}")
            continue
    
    return frequencies

def print_metrics_report(metrics):
    """Prints a formatted report of the metrics"""
    print("\n====== EVALUATION METRICS REPORT ======")
    
    for category in ['instruments', 'actions']:
        print(f"\n{category.upper()}:")
        print("=" * 80)
        print(f"{'Label':15s} {'F1-Score':>10s} {'Precision':>10s} {'Recall':>10s} {'AP':>10s} {'Support':>10s}")
        print("-" * 80)
        
        for label, scores in metrics[category]['per_class'].items():
            print(f"{label:15s} {scores['f1_score']:10.4f} {scores['precision']:10.4f} "
                  f"{scores['recall']:10.4f} {scores['ap_score']:10.4f} {scores['support']:10d}")
        
        print("\nMean Metrics:")
        means = metrics[category]['mean_metrics']
        print(f"Mean F1-Score:  {means['mean_f1']:.4f}")
        print(f"Mean Precision: {means['mean_precision']:.4f}")
        print(f"Mean Recall:    {means['mean_recall']:.4f}")
        print(f"Mean AP:        {means['mean_ap']:.4f}")
        print("-" * 80)

def main():
    """Compare ground truth and model predictions for all videos in HeiChole dataset"""
    try:
        # Initialize ModelLoader
        loader = ModelLoader()
        
        # Load models
        yolo_model = loader.load_yolo_model()
        verb_model = loader.load_verb_model()
        dataset_dir = str(loader.dataset_path)
        
        # Get all video IDs from the Labels directory
        labels_dir = Path(dataset_dir) / "Labels"
        videos_to_analyze = [f.stem for f in labels_dir.glob("*.json")]
        print(f"\nFound {len(videos_to_analyze)} videos to analyze: {', '.join(videos_to_analyze)}")
        
        # Für Test mit einem Video:
        #videos_to_analyze = ["VID13"]  
        #print(f"\nTesting with video: {videos_to_analyze[0]}")

        print("\n==========================================")
        print("GROUND TRUTH ANALYSIS")
        print("==========================================")
        
        # Ground Truth Analysis
        gt_distribution = analyze_label_distribution(dataset_dir, videos_to_analyze)
        
        print("\n==========================================")
        print("MODEL PREDICTIONS ANALYSIS")
        print("==========================================")
        
        # Create evaluator
        evaluator = HeiCholeEvaluator(
            yolo_model=yolo_model, 
            verb_model=verb_model, 
            dataset_dir=dataset_dir
        )
        
        # Initialize Binary Metrics Calculator
        metrics_calculator = BinaryMetricsCalculator(confidence_threshold=0.0001)
        
        # Collect Predictions and Ground Truth
        predictions_per_frame = {}
        ground_truth = {}
        
        # Process each video
        for video in videos_to_analyze:
            # Load Ground Truth
            gt = evaluator.load_ground_truth(video)
            ground_truth.update(gt)
            
            # Process each frame
            video_folder = os.path.join(dataset_dir, "Videos", video)
            for frame_file in os.listdir(video_folder):
                if frame_file.endswith('.png'):
                    frame_id = f"{video}_frame{frame_file.split('.')[0]}"
                    img_path = os.path.join(video_folder, frame_file)
                    
                    # Get Frame Predictions
                    frame_predictions = evaluator.evaluate_frame(
                        img_path,
                        gt[int(frame_file.split('.')[0])],
                        save_visualization=False
                    )
                    
                    if frame_predictions:
                        predictions_per_frame[frame_id] = frame_predictions
        
        # Calculate metrics
        binary_metrics = metrics_calculator.calculate_metrics(predictions_per_frame, ground_truth)
        
        # Calculate total frames
        total_frames = sum(sum(1 for _ in (Path(dataset_dir) / "Videos" / video).glob("*.png")) 
                         for video in videos_to_analyze)
        
        # Print results
        print("\n==========================================")
        print("FINAL COMPARISON")
        print("==========================================")
        
        # Print Ground Truth Distribution
        print("\nGROUND TRUTH DISTRIBUTION:")
        print("=" * 70)
        print(f"{'Category':20s} {'Count':>10s} {'Percentage':>12s}")
        print("-" * 70)
        
        # Print Instrument Statistics
        print("\nINSTRUMENTS:")
        for instr, count in sorted(gt_distribution['instruments'].items()):
            percentage = (count / total_frames) * 100
            print(f"{instr:20s} {count:10d} {percentage:11.2f}%")
        
        # Print Action Statistics
        print("\nACTIONS:")
        for action, count in sorted(gt_distribution['actions'].items()):
            percentage = (count / total_frames) * 100
            print(f"{action:20s} {count:10d} {percentage:11.2f}%")
        
        # Print Model Predictions and Metrics
        print("\nMODEL PREDICTIONS WITH METRICS:")
        print("=" * 90)
        print(f"{'Category':20s} {'Predictions':>12s} {'AP Score':>10s} {'F1 Score':>10s} {'Precision':>10s} {'Recall':>10s}")
        print("-" * 90)
        
        # Print Instrument Metrics
        print("\nINSTRUMENTS:")
        for instr, metrics in sorted(binary_metrics['instruments']['per_class'].items()):
            pred_count = metrics['predictions']
            ap = metrics['ap_score']
            f1 = metrics['f1_score']
            precision = metrics['precision']
            recall = metrics['recall']
            
            percentage = (pred_count / total_frames) * 100
            print(f"{instr:20s} {pred_count:10d} ({percentage:6.2f}%) {ap:8.4f} {f1:8.4f} {precision:8.4f} {recall:8.4f}")
        
        # Print Action Metrics
        print("\nACTIONS:")
        for action, metrics in sorted(binary_metrics['actions']['per_class'].items()):
            pred_count = metrics['predictions']
            ap = metrics['ap_score']
            f1 = metrics['f1_score']
            precision = metrics['precision']
            recall = metrics['recall']
            
            percentage = (pred_count / total_frames) * 100
            print(f"{action:20s} {pred_count:10d} ({percentage:6.2f}%) {ap:8.4f} {f1:8.4f} {precision:8.4f} {recall:8.4f}")
        
        # Print Mean Scores
        print("\nMEAN SCORES:")
        print("=" * 50)
        print("INSTRUMENTS:")
        print(f"mAP: {binary_metrics['instruments']['mean_metrics']['mean_ap']:.4f}")
        print(f"F1:  {binary_metrics['instruments']['mean_metrics']['mean_f1']:.4f}")
        print("\nACTIONS:")
        print(f"mAP: {binary_metrics['actions']['mean_metrics']['mean_ap']:.4f}")
        print(f"F1:  {binary_metrics['actions']['mean_metrics']['mean_f1']:.4f}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
