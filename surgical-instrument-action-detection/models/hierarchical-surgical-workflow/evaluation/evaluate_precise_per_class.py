import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import average_precision_score
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
#from verb_recognition.models.Backbone_SurgicalActionNet import SurgicalVerbRecognition


# Constants
#CONFIDENCE_THRESHOLD = 0.6
CONFIDENCE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.3
VIDEOS_TO_ANALYZE = ["VID92", "VID96", "VID103", "VID110", "VID111"]
#VIDEOS_TO_ANALYZE = ["VID92"]

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


class ModelLoader:
    def __init__(self):
        self.hierarchical_dir = hierarchical_dir
        self.setup_paths()

    def setup_paths(self):
        """Defines all important paths for the models"""
        # YOLO model path
        #self.yolo_weights = self.hierarchical_dir / "Instrument-classification-detection/weights/instrument_detector/best_v35.pt"
        self.yolo_weights = self.hierarchical_dir / "Instrument-classification-detection/weights/instrument_detector/weights/epoch0.pt"

        # Verb model path
        self.verb_model_path = self.hierarchical_dir / "verb_recognition/checkpoints/jumping-tree-47/last.ckpt"
        #self.verb_model_path = self.hierarchical_dir / "verb_recognition/checkpoints/genial-eon-54/last.ckpt"
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

    def evaluate_frame(self, img_path, ground_truth, save_visualization=False):
        """
        Evaluates a single frame using hierarchical recognition:
        1. YOLO detects instruments
        2. For each detected instrument, predict the surgical verb
        3. Create instrument-verb pairs
        
        Args:
            img_path: Path to the frame image
            ground_truth: Ground truth annotations for this frame
            save_visualization: Whether to save visualization of detections
        
        Returns:
            List of predictions, each containing:
            [video_frame, instrument, verb, instrument_verb_pair, instrument_conf, verb_conf]
        """
        # Initialize predictions list and extract frame information
        frame_predictions = []
        frame_number = int(os.path.basename(img_path).split('.')[0])
        video_name = os.path.basename(os.path.dirname(img_path))
        
        # Load and prepare image for visualization
        img = Image.open(img_path)
        original_img = img.copy()
        draw = ImageDraw.Draw(original_img)
        try:
            # Try to load a nice font, fallback to default if not available
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        try:
            # Step 1: Instrument Detection using YOLO
            yolo_results = self.yolo_model(img, verbose=False)
            valid_detections = []
            
            # Process each YOLO detection
            for detection in yolo_results[0].boxes:
                instrument_class = int(detection.cls)
                confidence = float(detection.conf)
                
                # Only consider predictions above confidence threshold
                if instrument_class < 6 and confidence >= CONFIDENCE_THRESHOLD:
                    instrument_name = TOOL_MAPPING[instrument_class]
                    
                    # Store valid detection
                    valid_detections.append({
                        'class': instrument_class,
                        'confidence': confidence,
                        'box': detection.xyxy[0],
                        'name': instrument_name
                    })
            
            # Sort detections by confidence (process high confidence detections first)
            valid_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Step 2: Process each detected instrument
            for idx, detection in enumerate(valid_detections):
                instrument_name = detection['name']
                box = detection['box']
                confidence = detection['confidence']
                
                # Crop image region with the detected instrument
                x1, y1, x2, y2 = map(int, box)
                instrument_crop = img.crop((x1, y1, x2, y2))
                
                # Prepare cropped image for verb model
                crop_tensor = self.transform(instrument_crop).unsqueeze(0).to(self.device)
                
                # Step 3: Predict verb for the detected instrument
                verb_outputs = self.verb_model(crop_tensor, [instrument_name])
                verb_probs = verb_outputs['probabilities']
                
                # Get top 3 verb predictions for this instrument
                top_verbs = []
                
                # Process top 3 verb predictions
                for verb_model_idx in torch.topk(verb_probs[0], k=3).indices.cpu().numpy():
                    try:
                        # Map model index to evaluation index
                        eval_verb_idx = VERB_MODEL_TO_EVAL_MAPPING[verb_model_idx]
                        verb_name = VERB_MAPPING[eval_verb_idx]
                        verb_prob = float(verb_probs[0][verb_model_idx])
                        
                        # Only consider valid instrument-verb combinations
                        if (verb_name in self.VALID_PAIRS[instrument_name]):
                            top_verbs.append({
                                'name': verb_name,
                                'probability': verb_prob
                            })
                    except KeyError as e:
                        print(f"Warning: Unexpected verb model index {verb_model_idx}")
                
                # Step 4: Create prediction and visualization for best verb
                if top_verbs:
                    # Get verb with highest probability
                    best_verb = max(top_verbs, key=lambda x: x['probability'])
                    verb_name = best_verb['name']
                    verb_conf = best_verb['probability']
                    pair = f"{instrument_name}_{verb_name}"
                    
                    # Format: [frame_id, instrument, verb, pair, instrument_conf, verb_conf]
                    prediction = [
                        f"{video_name}_frame{frame_number}",
                        instrument_name,
                        verb_name,
                        pair,
                        confidence,    # YOLO confidence score
                        verb_conf     # Verb model confidence score
                    ]
                    frame_predictions.append(prediction)
                    
                    # Draw bounding box and predictions
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                    text_color = 'blue' if confidence >= CONFIDENCE_THRESHOLD else 'orange'
                    draw.text((x1, y1-25), 
                            f"Pred: {instrument_name}-{verb_name}\n"
                            f"Conf: {confidence:.2f}, Verb: {verb_conf:.2f}", 
                            fill=text_color, font=font)
            
            # Step 5: Add ground truth visualization
            img_width, img_height = original_img.size
            gt_y_start = img_height - 150  # Start position for ground truth text
            
            # Draw background for ground truth
            draw.rectangle([10, gt_y_start, img_width-10, img_height-10], 
                        fill='white', outline='black')
            
            # Write ground truth heading and pairs
            draw.text((20, gt_y_start + 5), "Ground Truth:", fill='black', font=font)
            
            y_pos = gt_y_start + 35
            for pair, count in ground_truth['pairs'].items():
                if count > 0:
                    draw.text((20, y_pos), f"GT: {pair}", fill='green', font=small_font)
                    y_pos += 20
            
            # Save visualization if requested
            if save_visualization:
                viz_dir = os.path.join("/data/Bartscht/VID92_val", "visualizations")
                os.makedirs(viz_dir, exist_ok=True)
                save_path = os.path.join(viz_dir, f"{video_name}_frame{frame_number}.png")
                original_img.save(save_path)
            
            return frame_predictions
                
        except Exception as e:
            print(f"Error processing frame {frame_number}: {str(e)}")
            return []

    def evaluate(self):
        """
        Comprehensive evaluation of model performance across all specified videos.
        Ensures all possible classes are evaluated, even if they're never predicted.
        
        The evaluation process:
        1. Initializes tracking for ALL possible classes (not just predicted ones)
        2. Processes each video frame by frame
        3. Updates metrics and predictions
        4. Calculates final metrics including:
            - Traditional metrics (Precision, Recall, F1)
            - Average Precision (AP) for ALL classes
            - mean Average Precision (mAP) across all classes
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Initialize collection structures for ALL possible classes
        all_predictions = {
            'instruments': {instr: [] for instr in TOOL_MAPPING.values()},
            'verbs': {verb: [] for verb in VERB_MAPPING.values()},
            'pairs': {}
        }
        
        # Initialize all possible instrument-verb pairs based on VALID_PAIRS
        for instrument, valid_verbs in self.VALID_PAIRS.items():
            for verb in valid_verbs:
                pair = f"{instrument}_{verb}"
                all_predictions['pairs'][pair] = []
        
        # Traditional metrics counters
        results = {
            'instruments': {'TP': 0, 'FP': 0, 'FN': 0},
            'verbs': {'TP': 0, 'FP': 0, 'FN': 0},
            'pairs': {'TP': 0, 'FP': 0, 'FN': 0}
        }
        
        # Track occurrences of each class in ground truth
        ground_truth_occurrences = {
            'instruments': defaultdict(int),
            'verbs': defaultdict(int),
            'pairs': defaultdict(int)
        }
        
        # Process each video in the evaluation set
        for video in VIDEOS_TO_ANALYZE:
            # Load ground truth annotations
            ground_truth = self.load_ground_truth(video)
            
            # Get frame files
            video_folder = os.path.join(self.dataset_dir, "videos", video)
            frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.png')])
            
            # Process each frame
            for frame_file in tqdm(frame_files, desc=f"Evaluating {video}"):
                frame_number = int(frame_file.split('.')[0])
                img_path = os.path.join(video_folder, frame_file)
                
                # Get frame predictions
                frame_predictions = self.evaluate_frame(
                    img_path,
                    ground_truth[frame_number],
                    save_visualization=True
                )
                
                # Get ground truth for this frame
                gt_frame = ground_truth[frame_number]
                
                # Update ground truth occurrence counters
                for instr, count in gt_frame['instruments'].items():
                    if count > 0:
                        ground_truth_occurrences['instruments'][instr] += 1
                for verb, count in gt_frame['verbs'].items():
                    if count > 0:
                        ground_truth_occurrences['verbs'][verb] += 1
                for pair, count in gt_frame['pairs'].items():
                    if count > 0:
                        ground_truth_occurrences['pairs'][pair] += 1
                
                # Track matched predictions
                matched = {
                    'instruments': set(),
                    'verbs': set(),
                    'pairs': set()
                }
                
                # Process frame predictions
                for pred in frame_predictions:
                    _, instrument, verb, pair, instrument_conf, verb_conf = pred
                    pair_conf = min(instrument_conf, verb_conf)
                    
                    # Update predictions for AP calculation
                    # For instruments
                    all_predictions['instruments'][instrument].append({
                        'confidence': instrument_conf,
                        'ground_truth': 1 if gt_frame['instruments'][instrument] > 0 else 0
                    })
                    
                    # For verbs
                    all_predictions['verbs'][verb].append({
                        'confidence': verb_conf,
                        'ground_truth': 1 if gt_frame['verbs'][verb] > 0 else 0
                    })
                    
                    # For pairs
                    all_predictions['pairs'][pair].append({
                        'confidence': pair_conf,
                        'ground_truth': 1 if gt_frame['pairs'][pair] > 0 else 0
                    })
                    
                    # Update traditional metrics
                    self._update_traditional_metrics(
                        results, matched, gt_frame,
                        instrument, verb, pair
                    )
                
                # Count False Negatives
                self._count_false_negatives(results, matched, gt_frame)
        
        # Calculate and format final metrics
        final_metrics = self._calculate_final_metrics(
            results, all_predictions, ground_truth_occurrences
        )
        
        # Print detailed evaluation results
        self._print_evaluation_results(final_metrics)
        
        return final_metrics

    def _update_traditional_metrics(self, results, matched, gt_frame, instrument, verb, pair):
        """
        Updates the traditional metrics (TP, FP) for a single prediction.
        
        Args:
            results: Dictionary containing current metrics
            matched: Set of already matched predictions
            gt_frame: Ground truth for current frame
            instrument, verb, pair: Current prediction elements
        """
        # Instruments
        if gt_frame['instruments'][instrument] > 0:
            if instrument not in matched['instruments']:
                results['instruments']['TP'] += 1
                matched['instruments'].add(instrument)
        else:
            results['instruments']['FP'] += 1
        
        # Verbs
        if gt_frame['verbs'][verb] > 0:
            if verb not in matched['verbs']:
                results['verbs']['TP'] += 1
                matched['verbs'].add(verb)
        else:
            results['verbs']['FP'] += 1
        
        # Pairs
        if gt_frame['pairs'][pair] > 0:
            if pair not in matched['pairs']:
                results['pairs']['TP'] += 1
                matched['pairs'].add(pair)
        else:
            results['pairs']['FP'] += 1

    def _count_false_negatives(self, results, matched, gt_frame):
        """
        Counts false negatives for all categories in current frame.
        
        Args:
            results: Dictionary containing current metrics
            matched: Set of already matched predictions
            gt_frame: Ground truth for current frame
        """
        # Instruments FN
        results['instruments']['FN'] += sum(
            1 for inst, count in gt_frame['instruments'].items()
            if count > 0 and inst not in matched['instruments']
        )
        
        # Verbs FN
        results['verbs']['FN'] += sum(
            1 for verb, count in gt_frame['verbs'].items()
            if count > 0 and verb not in matched['verbs']
        )
        
        # Pairs FN
        results['pairs']['FN'] += sum(
            1 for pair, count in gt_frame['pairs'].items()
            if count > 0 and pair not in matched['pairs']
        )

    def _calculate_per_class_metrics(self, all_predictions, gt_occurrences):
        """
        Calculate detailed per-class metrics similar to the multitask model.
        
        Args:
            all_predictions: Dictionary with predictions for each class
            gt_occurrences: Count of ground truth occurrences per class
        
        Returns:
            Dictionary with detailed metrics for each class
        """
        per_class_metrics = {
            'instruments': {},
            'verbs': {},
            'pairs': {}
        }
        
        # Calculate total support for distribution calculation
        total_supports = {
            'instruments': sum(gt_occurrences['instruments'].values()),
            'verbs': sum(gt_occurrences['verbs'].values()),
            'pairs': sum(gt_occurrences['pairs'].values())
        }
        
        # Calculate metrics for each class
        for category in ['instruments', 'verbs', 'pairs']:
            for class_name, predictions in all_predictions[category].items():
                support = gt_occurrences[category].get(class_name, 0)
                
                # If there are no predictions, set default values
                if not predictions:
                    per_class_metrics[category][class_name] = {
                        'AP': 0.0,
                        'support': support,
                        'dist': (support / total_supports[category] * 100) if total_supports[category] > 0 else 0,
                        'pred': 0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0
                    }
                    continue
                
                # Extract ground truth and prediction scores
                y_true = np.array([p['ground_truth'] for p in predictions])
                y_scores = np.array([p['confidence'] for p in predictions])
                
                # Apply 0.5 threshold for traditional metrics
                y_pred = (y_scores >= 0.5).astype(int)
                n_pred = np.sum(y_pred)
                
                # Calculate traditional metrics
                tp = np.sum((y_pred == 1) & (y_true == 1))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                fn = np.sum((y_pred == 0) & (y_true == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # Calculate AP
                try:
                    ap = average_precision_score(y_true, y_scores) if support > 0 else 0.0
                except:
                    ap = 0.0
                
                # Store metrics
                per_class_metrics[category][class_name] = {
                    'AP': ap,
                    'support': support,
                    'dist': (support / total_supports[category] * 100) if total_supports[category] > 0 else 0,
                    'pred': n_pred,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
        
        return per_class_metrics

    def _calculate_final_metrics(self, results, all_predictions, gt_occurrences):
        """
        Calculate final metrics for the evaluation.
        
        Args:
            results: Dictionary with traditional metrics (TP, FP, FN)
            all_predictions: Dictionary with all class predictions
            gt_occurrences: Count of ground truth occurrences per class
        
        Returns:
            Dictionary with all calculated metrics
        """
        final_metrics = {}
        
        # Calculate per-class metrics
        per_class_metrics = self._calculate_per_class_metrics(all_predictions, gt_occurrences)
        
        for category in ['instruments', 'verbs', 'pairs']:
            # Calculate traditional overall metrics
            TP = results[category]['TP']
            FP = results[category]['FP']
            FN = results[category]['FN']
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate mean metrics from per-class metrics
            valid_classes = [m for c, m in per_class_metrics[category].items() if m['support'] > 0]
            
            mean_ap = np.mean([m['AP'] for m in valid_classes]) if valid_classes else 0.0
            mean_f1 = np.mean([m['f1'] for m in valid_classes]) if valid_classes else 0.0
            
            # Calculate weighted means
            weights = [m['support'] for m in valid_classes]
            weighted_ap = np.average([m['AP'] for m in valid_classes], weights=weights) if valid_classes else 0.0
            weighted_f1 = np.average([m['f1'] for m in valid_classes], weights=weights) if valid_classes else 0.0
            
            # Store all metrics
            final_metrics[category] = {
                'traditional': {
                    'TP': TP, 'FP': FP, 'FN': FN,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                },
                'per_class': per_class_metrics[category],
                'mean': {
                    'AP': mean_ap,
                    'f1': mean_f1
                },
                'weighted_mean': {
                    'AP': weighted_ap,
                    'f1': weighted_f1
                }
            }
        
        return final_metrics

    def _print_evaluation_results(self, final_metrics):
        """
        Print detailed evaluation results in a format similar to the multitask model.
        
        Args:
            final_metrics: Dictionary with all calculated metrics
        """
        print("\nEvaluation Results:")
        print("="*80)
        
        for category in ['instruments', 'verbs', 'pairs']:
            print(f"\n{category.upper()}:")
            print("-"*80)
            print(f"{'Item':<20} {'Support':>8} {'Dist%':>8} {'Pred':>8} {'AP':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
            print("-"*80)
            
            # Print metrics for each class
            for class_name, metrics in sorted(final_metrics[category]['per_class'].items()):
                print(f"{class_name:<20} {metrics['support']:>8d} {metrics['dist']:>8.2f} {metrics['pred']:>8d} "
                     f"{metrics['AP']:>8.4f} {metrics['f1']:>8.4f} {metrics['precision']:>8.4f} {metrics['recall']:>8.4f}")
            
            # Print mean metrics
            print("-"*80)
            print(f"Mean         {'-':>8} {'-':>8} {'-':>8} "
                 f"{final_metrics[category]['mean']['AP']:>8.4f} {final_metrics[category]['mean']['f1']:>8.4f}")
            print(f"Weighted Mean{'-':>8} {'-':>8} {'-':>8} "
                 f"{final_metrics[category]['weighted_mean']['AP']:>8.4f} {final_metrics[category]['weighted_mean']['f1']:>8.4f}")
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
            if os.path.isdir(os.path.join(videos_dir, video)):
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
        
        # Optional: Save results to file
        try:
            output_dir = os.path.join(current_dir, "evaluation_results")
            os.makedirs(output_dir, exist_ok=True)
            
            import pickle
            with open(os.path.join(output_dir, "hierarchical_eval_results.pkl"), "wb") as f:
                pickle.dump(results, f)
            print(f"Results saved to {output_dir}/hierarchical_eval_results.pkl")
            
            # Save metrics summary as CSV
            csv_summary = []
            csv_summary.append("Category,Class,Support,DistPct,Pred,AP,F1,Precision,Recall")
            
            for category in ['instruments', 'verbs', 'pairs']:
                for class_name, metrics in sorted(results[category]['per_class'].items()):
                    csv_summary.append(f"{category},{class_name},{metrics['support']},{metrics['dist']:.2f},"
                                     f"{metrics['pred']},{metrics['AP']:.4f},{metrics['f1']:.4f},"
                                     f"{metrics['precision']:.4f},{metrics['recall']:.4f}")
                
                # Add mean and weighted mean
                csv_summary.append(f"{category},Mean,,,,"
                               f"{results[category]['mean']['AP']:.4f},{results[category]['mean']['f1']:.4f},,")
                csv_summary.append(f"{category},WeightedMean,,,,"
                               f"{results[category]['weighted_mean']['AP']:.4f},{results[category]['weighted_mean']['f1']:.4f},,")
                csv_summary.append("")  # Empty line between categories
            
            with open(os.path.join(output_dir, "hierarchical_eval_metrics.csv"), "w") as f:
                f.write("\n".join(csv_summary))
            print(f"Metrics saved to {output_dir}/hierarchical_eval_metrics.csv")
            
        except Exception as e:
            print(f"Warning: Could not save results to file: {str(e)}")
        
    except Exception as e:
        print(f"❌ Error during initialization or evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()