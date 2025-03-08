import os
import sys
from pathlib import Path
import torch
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
import json
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO
import pytorch_lightning as pl

# Path configuration - Adjust to your environment
current_dir = Path(__file__).resolve().parent
hierarchical_dir = current_dir.parent
sys.path.append(str(hierarchical_dir))

# Import the trained SurgicalActionRecognition model
from verb_recognition.models.pointer_IV_predictor import SurgicalActionRecognition

# Constants
CONFIDENCE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.3
VIDEOS_TO_ANALYZE = ["VID92", "VID96", "VID103", "VID110", "VID111"]

# Mappings
TOOL_MAPPING = {
    0: 'grasper', 1: 'bipolar', 2: 'hook', 
    3: 'scissors', 4: 'clipper', 5: 'irrigator'
}

VERB_MAPPING = {
    0: 'grasp', 1: 'retract', 2: 'dissect', 3: 'coagulate', 
    4: 'clip', 5: 'cut', 6: 'aspirate', 7: 'irrigate', 
    8: 'pack', 9: 'null_verb'
}

class ModelLoader:
    def __init__(self):
        self.hierarchical_dir = hierarchical_dir
        self.setup_paths()

    def setup_paths(self):
        """Defines all important paths for the models"""
        # YOLO model path
        self.yolo_weights = self.hierarchical_dir / "Instrument-classification-detection/weights/instrument_detector/best_v35.pt"
        # SurgicalActionRecognition model path
        self.action_model_path = self.hierarchical_dir / Path("verb_recognition/checkpoints/leafy-gorge-5/last.ckpt") 
        self.dataset_path = Path("/data/Bartscht/CholecT50")
        
        print(f"YOLO weights path: {self.yolo_weights}")
        print(f"Action model path: {self.action_model_path}")
        print(f"Dataset path: {self.dataset_path}")

        # Validate paths
        if not self.yolo_weights.exists():
            raise FileNotFoundError(f"YOLO weights not found at: {self.yolo_weights}")
        if not self.action_model_path.exists():
            raise FileNotFoundError(f"Action model checkpoint not found at: {self.action_model_path}")
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.dataset_path}")

    def load_yolo_model(self):
        try:
            model = YOLO(str(self.yolo_weights))
            print("YOLO model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading YOLO model: {str(e)}")
            raise

    def load_action_model(self):
        try:
            model = SurgicalActionRecognition.load_from_checkpoint(
                checkpoint_path=str(self.action_model_path)
            )
            model.eval()
            print("SurgicalActionRecognition model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading action model: {str(e)}")
            raise


class IntegratedEvaluator:
    def __init__(self, yolo_model, action_model, dataset_dir):
        """
        Initializes the IntegratedEvaluator.
        
        Args:
            yolo_model: Pretrained YOLO model for instrument detection
            action_model: Pretrained SurgicalActionRecognition model
            dataset_dir: CholecT50 dataset directory
        """
        # Device for computations (CUDA if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models
        self.yolo_model = yolo_model
        self.action_model = action_model.to(self.device)
        self.action_model.eval()  # Set model to evaluation mode
        
        # Path to dataset
        self.dataset_dir = dataset_dir
        
        # Transformations for image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Mappings between CholecT50 indices and names
        self.tool_mapping = TOOL_MAPPING
        self.tool_idx_to_name = self.tool_mapping
        self.tool_name_to_idx = {name: idx for idx, name in self.tool_mapping.items()}
        
        self.verb_mapping = VERB_MAPPING
        self.verb_idx_to_name = self.verb_mapping
        self.verb_name_to_idx = {name: idx for idx, name in self.verb_mapping.items()}
        
        # Initialize dictionary for iv_pairs_components
        self.iv_pairs_components = {}
        
        # Initialize IV-Pair Mapper for CholecT50 format
        self.initialize_iv_mapping()
        
        # Storage for predictions and ground truth
        self.reset_prediction_storage()
        
        print(f"Initialized evaluator with {len(self.model_to_cholect50)} instrument-verb pairs")

    def initialize_iv_mapping(self):
        """
        Initializes mappings between the model format for instrument-verb pairs
        and the CholecT50 evaluation format.
        """
        # The classes of the trained verb module
        model_action_classes = {
            'Hook-Dissect': 0,
            'Grasper-Retract': 1,
            'Bipolar-Coagulate': 2,
            'Grasper-Grasp': 3,
            'Clipper-Clip': 4,
            'Hook_null-Verb': 5,
            'Irrigator-Aspirate': 6,
            'Scissors-Cut': 7,
            'Grasper_null-Verb': 8,
            'Bipolar-Dissect': 9,
            'Hook-Coagulate': 10,
            'Hook-Retract': 11,
            'Irrigator_null-Verb': 12,
            'Bipolar_null-Verb': 13,
            'Irrigator-Retract': 14,
            'Irrigator-Irrigate': 15,
            'Clipper_null-Verb': 16,
            'Bipolar-Retract': 17,
            'Irrigator-Dissect': 18,
            'Scissors_null-Verb': 19,
            'Scissors-Dissect': 20,
            'Bipolar-Grasp': 21,
            'Grasper-Dissect': 22,
            'Grasper-Pack': 23,
            'Scissors-Coagulate': 24
        }
        
        # We use these classes directly instead of loading from the model
        self.action_classes = model_action_classes
        
        # Mapping from model indices to pair names
        self.idx_to_action = {v: k for k, v in self.action_classes.items()}
        
        # Mapping from model pair names to CholecT50 representation
        self.model_to_cholect50 = {}
        
        # Create mapping from model format to CholecT50 format
        for model_pair_name, model_idx in self.action_classes.items():
            # Split the string to extract instrument and verb
            if "_null-Verb" in model_pair_name:
                instrument = model_pair_name.split("_null-Verb")[0].lower()
                verb = "null_verb"
            else:
                parts = model_pair_name.split('-')
                instrument = parts[0].lower()
                verb = parts[1].lower()
            
            # Create the CholecT50 key
            cholect50_key = f"{instrument}_{verb}"
            
            # Store the mapping
            self.model_to_cholect50[model_pair_name] = cholect50_key
            
            # Also store components for later use
            self.iv_pairs_components[model_pair_name] = {
                'instrument': instrument,
                'verb': verb,
                'instrument_idx': self.tool_name_to_idx.get(instrument, -1),
                'verb_idx': self.verb_name_to_idx.get(verb, -1)
            }
        
        # Collect all unique instruments and verbs
        self.unique_instruments = set(comp['instrument'] for comp in self.iv_pairs_components.values())
        self.unique_verbs = set(comp['verb'] for comp in self.iv_pairs_components.values())
        self.unique_pairs = set(self.model_to_cholect50.values())
        
        print(f"Initialized {len(self.action_classes)} instrument-verb pairs")
        print(f"Unique instruments: {len(self.unique_instruments)}")
        print(f"Unique verbs: {len(self.unique_verbs)}")
        print(f"Unique CholecT50 pairs: {len(self.unique_pairs)}")

    def reset_prediction_storage(self):
        """Resets storage for predictions and ground truth"""
        # Initialize storage structure for predictions and ground truth
        self.all_predictions = {
            'instruments': defaultdict(list),  # Format: {class_name: [(pred_conf, gt_label), ...]}
            'verbs': defaultdict(list),
            'iv_pairs': defaultdict(list)
        }
        
        # For tracking frame-level predictions
        self.frame_predictions = []

    def load_ground_truth(self, video):
        """
        Loads ground truth annotations for a specific video.
        
        Args:
            video: Video identifier (e.g., "VID92")
        Returns:
            Dictionary with frame annotations
        """
        # Path to label files
        labels_folder = os.path.join(self.dataset_dir, "labels")
        json_file = os.path.join(labels_folder, f"{video}.json")
        
        # Defaultdict for frame annotations
        frame_annotations = defaultdict(lambda: {
            'instruments': defaultdict(int),
            'verbs': defaultdict(int),
            'iv_pairs': defaultdict(int)
        })
        
        # Load JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)
            annotations = data['annotations']
            
            # Process annotations
            for frame, instances in annotations.items():
                frame_number = int(frame)
                for instance in instances:
                    instrument_idx = instance[1]
                    verb_idx = instance[7]
                    
                    # Validate and map instrument
                    if isinstance(instrument_idx, int) and 0 <= instrument_idx < 6:
                        instrument_name = self.tool_idx_to_name[instrument_idx]
                        frame_annotations[frame_number]['instruments'][instrument_name] += 1
                        
                        # Validate and map verb
                        if isinstance(verb_idx, int) and 0 <= verb_idx < 10:
                            verb_name = self.verb_idx_to_name[verb_idx]
                            frame_annotations[frame_number]['verbs'][verb_name] += 1
                            
                            # Create instrument-verb pair in CholecT50 format
                            iv_key = f"{instrument_name}_{verb_name}"
                            frame_annotations[frame_number]['iv_pairs'][iv_key] += 1
        
        return frame_annotations

    def evaluate_frame(self, img_path, gt_frame, save_visualization=False):
        """
        Evaluates a single frame with integrated recognition:
        1. YOLO detects instruments (only bounding boxes)
        2. For each detected instrument, a prediction with the SurgicalActionRecognition model
        
        Args:
            img_path: Path to the frame image
            gt_frame: Ground truth annotations for this frame
            save_visualization: Whether to save a visualization of the detections
        
        Returns:
            List of predictions for this frame
        """
        # Initialize prediction list and extract frame information
        frame_predictions = []
        frame_number = int(os.path.basename(img_path).split('.')[0])
        video_name = os.path.basename(os.path.dirname(img_path))
        
        # Load and prepare image for visualization
        img = Image.open(img_path)
        original_img = img.copy()
        draw = ImageDraw.Draw(original_img)
        
        try:
            # Load font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        try:
            # Step 1: Instrument detection with YOLO (only bounding boxes)
            yolo_results = self.yolo_model(img, verbose=False)
            valid_detections = []
            
            # Process each YOLO detection
            for detection in yolo_results[0].boxes:
                confidence = float(detection.conf)
                
                # Consider only predictions above the confidence threshold
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Store valid detection
                    valid_detections.append({
                        'confidence': confidence,
                        'box': detection.xyxy[0]
                    })
            
            # Sort detections by confidence
            valid_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Step 2: Process each detected region with the Action model
            for idx, detection in enumerate(valid_detections):
                box = detection.get('box')
                yolo_confidence = detection.get('confidence')
                
                # Crop image region with the detected instrument
                x1, y1, x2, y2 = map(int, box)
                instrument_crop = img.crop((x1, y1, x2, y2))
                
                # Prepare cropped image for the Action model
                crop_tensor = self.transform(instrument_crop).unsqueeze(0).to(self.device)
                
                # Step 3: Prediction with the SurgicalActionRecognition model
                with torch.no_grad():
                    output = self.action_model.predict_crop(crop_tensor)
                
                # Extract information from the prediction
                action_idx = output['action_idx']
                action_name = output['action_name']
                
                # Convert to CholecT50 format
                cholect50_key = self.model_to_cholect50.get(action_name)
                if not cholect50_key:
                    print(f"Warning: Could not convert model format '{action_name}' to CholecT50 format")
                    continue
                
                # Split the CholecT50 key
                instrument, verb = cholect50_key.split('_', 1)
                pair_confidence = output['confidence']
                
                # Create prediction entry
                prediction = {
                    'frame_id': f"{video_name}_frame{frame_number}",
                    'instrument': instrument,
                    'verb': verb,
                    'iv_pair': cholect50_key,
                    'model_action_name': action_name,
                    'action_idx': action_idx,
                    'yolo_confidence': yolo_confidence,
                    'pair_confidence': pair_confidence,
                    'box': (x1, y1, x2, y2)
                }
                frame_predictions.append(prediction)
                
                # Draw bounding box and predictions
                if save_visualization:
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                    text_color = 'blue' if yolo_confidence >= CONFIDENCE_THRESHOLD else 'orange'
                    draw.text((x1, y1-25), 
                             f"Pred: {instrument}-{verb}\n"
                             f"Conf: {yolo_confidence:.2f}, Verb: {pair_confidence:.2f}", 
                             fill=text_color, font=font)
            
            # Add ground truth visualization
            if save_visualization:
                img_width, img_height = original_img.size
                gt_y_start = img_height - 150  # Start position for ground truth text
                
                # Draw background for ground truth
                draw.rectangle([10, gt_y_start, img_width-10, img_height-10], 
                            fill='white', outline='black')
                
                # Write ground truth heading and pairs
                draw.text((20, gt_y_start + 5), "Ground Truth:", fill='black', font=font)
                
                y_pos = gt_y_start + 35
                for pair, count in gt_frame['iv_pairs'].items():
                    if count > 0:
                        draw.text((20, y_pos), f"GT: {pair}", fill='green', font=small_font)
                        y_pos += 20
                
                # Save visualization
                viz_dir = os.path.join("/data/Bartscht/VID92_val", "visualizations")
                os.makedirs(viz_dir, exist_ok=True)
                save_path = os.path.join(viz_dir, f"{video_name}_frame{frame_number}.png")
                original_img.save(save_path)
            
            return frame_predictions
                
        except Exception as e:
            print(f"Error processing frame {frame_number}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def process_frame_metrics(self, frame_predictions, gt_frame):
        """
        Processes metrics for a single frame
        
        Args:
            frame_predictions: List of predictions for the current frame
            gt_frame: Ground truth for the current frame
        """
        # Track what has been detected in this frame
        detected_items = {
            'instruments': set(),
            'verbs': set(),
            'iv_pairs': set()
        }
        
        # Process each prediction
        for pred in frame_predictions:
            # Extract predicted components
            instrument = pred['instrument']
            verb = pred['verb']
            iv_pair = pred['iv_pair']
            confidence_instr = pred['yolo_confidence']
            confidence_verb = pred['pair_confidence']
            confidence_pair = min(confidence_instr, confidence_verb)  # Conservative approach
            
            # Mark as detected
            detected_items['instruments'].add(instrument)
            detected_items['verbs'].add(verb)
            detected_items['iv_pairs'].add(iv_pair)
            
            # Get ground truth values (1 if present, 0 if not)
            gt_instrument = 1 if gt_frame['instruments'].get(instrument, 0) > 0 else 0
            gt_verb = 1 if gt_frame['verbs'].get(verb, 0) > 0 else 0
            gt_pair = 1 if gt_frame['iv_pairs'].get(iv_pair, 0) > 0 else 0
            
            # Store instrument prediction with ground truth
            self.all_predictions['instruments'][instrument].append((confidence_instr, gt_instrument))
            
            # Store verb prediction
            self.all_predictions['verbs'][verb].append((confidence_verb, gt_verb))
            
            # Store pair prediction
            self.all_predictions['iv_pairs'][iv_pair].append((confidence_pair, gt_pair))
        
        # Process false negatives (items in ground truth but not detected)
        # For instruments
        for instr, count in gt_frame['instruments'].items():
            if count > 0 and instr not in detected_items['instruments']:
                # Add as a missed detection (confidence 0, ground truth 1)
                self.all_predictions['instruments'][instr].append((0.0, 1))
        
        # For verbs
        for verb, count in gt_frame['verbs'].items():
            if count > 0 and verb not in detected_items['verbs']:
                self.all_predictions['verbs'][verb].append((0.0, 1))
        
        # For pairs
        for pair, count in gt_frame['iv_pairs'].items():
            if count > 0 and pair not in detected_items['iv_pairs']:
                self.all_predictions['iv_pairs'][pair].append((0.0, 1))

    def evaluate(self):
        """
        Comprehensive evaluation of model performance across all specified videos.
        
        The evaluation process:
        1. Initializes tracking for ALL possible classes
        2. Processes each video frame by frame
        3. Updates metrics and predictions
        4. Calculates final metrics including:
           - Traditional metrics (precision, recall, F1)
           - Average Precision (AP) for ALL classes
           - Mean Average Precision (mAP) across all classes
        
        Returns:
            Dictionary with all evaluation metrics
        """
        # Initialize collection structures for ALL possible classes
        self.reset_prediction_storage()
        
        # Track occurrences of each class in ground truth
        ground_truth_occurrences = {
            'instruments': defaultdict(int),
            'verbs': defaultdict(int),
            'iv_pairs': defaultdict(int)
        }
        
        print("\nStarting evaluation process...")
        
        # Process each video in the evaluation set
        for video in VIDEOS_TO_ANALYZE:
            print(f"\nProcessing {video}...")
            
            # Load ground truth annotations
            ground_truth = self.load_ground_truth(video)
            
            # Get frame files
            video_folder = os.path.join(self.dataset_dir, "videos", video)
            frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.png')])
            
            # Process each frame
            for frame_file in tqdm(frame_files, desc=f"Evaluating {video}"):
                frame_number = int(frame_file.split('.')[0])
                img_path = os.path.join(video_folder, frame_file)
                
                # Get ground truth for this frame
                gt_frame = ground_truth[frame_number]
                
                # Update ground truth occurrence counters
                for instr, count in gt_frame['instruments'].items():
                    if count > 0:
                        ground_truth_occurrences['instruments'][instr] += 1
                for verb, count in gt_frame['verbs'].items():
                    if count > 0:
                        ground_truth_occurrences['verbs'][verb] += 1
                for pair, count in gt_frame['iv_pairs'].items():
                    if count > 0:
                        ground_truth_occurrences['iv_pairs'][pair] += 1
                
                # Get frame predictions
                frame_predictions = self.evaluate_frame(
                    img_path,
                    gt_frame,
                    save_visualization=False  # Change to True for visualizations
                )
                
                # Process metrics for this frame
                self.process_frame_metrics(frame_predictions, gt_frame)
        
        # Calculate and format final metrics
        final_metrics = self._calculate_final_metrics(ground_truth_occurrences)
        
        # Print detailed evaluation results
        self._print_evaluation_results(final_metrics, ground_truth_occurrences)
        
        return final_metrics

    def _calculate_final_metrics(self, gt_occurrences):
        """
        Calculates final metrics including AP for all possible classes.
        
        Args:
            gt_occurrences: Count of ground truth occurrences per class
        
        Returns:
            Dictionary with all final metrics
        """
        final_metrics = {}
        
        for category in ['instruments', 'verbs', 'iv_pairs']:
            # Calculate traditional metrics and per-class AP
            category_metrics = {}
            
            for item_name, predictions in self.all_predictions[category].items():
                # Unzip predictions and ground truth labels
                if not predictions:
                    # No predictions for this class
                    category_metrics[item_name] = {
                        'AP': 0.0 if gt_occurrences[category][item_name] > 0 else None,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'TP': 0,
                        'FP': 0,
                        'FN': 0,
                        'gt_count': gt_occurrences[category][item_name]
                    }
                    continue
                
                # Sort by confidence for reliable AP calculation
                predictions.sort(key=lambda x: x[0], reverse=True)
                
                confidences, gt_labels = zip(*predictions)
                confidences = np.array(confidences)
                gt_labels = np.array(gt_labels)
                gt_count = gt_occurrences[category][item_name]
                
                # Convert confidences to binary predictions at threshold 0.5
                binary_preds = (confidences >= 0.5).astype(int)
                
                # Count TP, FP, FN
                TP = np.sum((binary_preds == 1) & (gt_labels == 1))
                FP = np.sum((binary_preds == 1) & (gt_labels == 0))
                FN = np.sum((binary_preds == 0) & (gt_labels == 1))
                
                # Calculate precision, recall, F1
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # Calculate AP if class exists in ground truth
                if gt_count > 0:
                    # Handle special case where all confidences are the same
                    if len(set(confidences)) == 1:
                        # If all predictions are correct, AP = 1.0
                        # If all are wrong, AP = 0.0
                        # Otherwise, AP = precision
                        ap = precision if TP > 0 else 0.0
                    else:
                        # Normal case: use scikit-learn's AP calculation
                        ap = average_precision_score(gt_labels, confidences)
                else:
                    ap = None  # Class never appears in ground truth
                
                # Store metrics for this class
                category_metrics[item_name] = {
                    'AP': ap,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'TP': int(TP),
                    'FP': int(FP),
                    'FN': int(FN),
                    'gt_count': gt_count
                }
            
            # Calculate aggregated metrics
            total_TP = sum(metrics['TP'] for metrics in category_metrics.values())
            total_FP = sum(metrics['FP'] for metrics in category_metrics.values())
            total_FN = sum(metrics['FN'] for metrics in category_metrics.values())
            
            # Calculate micro-averaged metrics
            micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
            micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
            
            # Calculate mAP (without None values)
            valid_aps = [metrics['AP'] for metrics in category_metrics.values() if metrics['AP'] is not None]
            map_score = np.mean(valid_aps) if valid_aps else 0.0
            
            # Store all metrics
            final_metrics[category] = {
                'micro_metrics': {
                    'TP': total_TP,
                    'FP': total_FP, 
                    'FN': total_FN,
                    'precision': micro_precision,
                    'recall': micro_recall,
                    'f1': micro_f1
                },
                'per_class': category_metrics,
                'mAP': map_score
            }
        
        return final_metrics

    def _print_evaluation_results(self, final_metrics, gt_occurrences):
        """
        Prints detailed evaluation results including class coverage analysis.
        
        Args:
            final_metrics: Dictionary with all calculated metrics
            gt_occurrences: Count of ground truth occurrences per class
        """
        print("\nFinal Evaluation Results:")
        print("========================")
        
        for category in ['instruments', 'verbs', 'iv_pairs']:
            print(f"\n{category.upper()} METRICS:")
            print("-" * 80)
            
            # Print micro-averaged metrics
            micro = final_metrics[category]['micro_metrics']
            print(f"Overall True Positives: {micro['TP']}")
            print(f"Overall False Positives: {micro['FP']}")
            print(f"Overall False Negatives: {micro['FN']}")
            print(f"Micro-Precision: {micro['precision']:.4f}")
            print(f"Micro-Recall: {micro['recall']:.4f}")
            print(f"Micro-F1: {micro['f1']:.4f}")
            print(f"mAP: {final_metrics[category]['mAP']:.4f}")
            
            # Print per-class metrics
            print("\nPer-class Metrics:")
            print(f"{'Class Name':<30} {'GT Count':>8} {'AP':>8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'TP':>5} {'FP':>5} {'FN':>5}")
            print("-" * 95)
            
            for class_name, metrics in final_metrics[category]['per_class'].items():
                gt_count = metrics['gt_count']
                if metrics['AP'] is None:
                    ap_str = "N/A"
                else:
                    ap_str = f"{metrics['AP']:.4f}" if metrics['AP'] > 0 else "0.0000"
                
                print(f"{class_name:<30} {gt_count:>8} {ap_str:>8} {metrics['precision']:>10.4f} "
                      f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['TP']:>5} {metrics['FP']:>5} {metrics['FN']:>5}")
            
            print("-" * 95)
            print("=" * 80)
            
    def save_results_to_file(self, final_metrics, output_file):
        """
        Saves evaluation results to a JSON file.
        
        Args:
            final_metrics: Dictionary with all calculated metrics
            output_file: Path to the output JSON file
        """
        # Convert NumPy values to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # Convert metrics to serializable format
        serializable_metrics = convert_to_serializable(final_metrics)
        
        # Add timestamp and model information
        results = {
            "timestamp": str(np.datetime64('now')),
            "videos_analyzed": VIDEOS_TO_ANALYZE,
            "metrics": serializable_metrics
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")

    def create_summary_visualizations(self, final_metrics, output_dir):
        """
        Creates summary visualizations of the evaluation results.
        
        Args:
            final_metrics: Dictionary with all calculated metrics
            output_dir: Directory to save visualizations
        """
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot mAP for each category
        def plot_map_comparison():
            categories = ['instruments', 'verbs', 'iv_pairs']
            map_values = [final_metrics[cat]['mAP'] for cat in categories]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(categories, map_values, color=['#3498db', '#2ecc71', '#e74c3c'])
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=12)
            
            plt.title('Mean Average Precision (mAP) per Category', fontsize=14)
            plt.ylabel('mAP', fontsize=12)
            plt.ylim(0, 1.0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'map_comparison.png'), dpi=300)
            plt.close()
        
        # Plot precision, recall, and F1 for each category
        def plot_metrics_comparison():
            categories = ['instruments', 'verbs', 'iv_pairs']
            precision_values = [final_metrics[cat]['micro_metrics']['precision'] for cat in categories]
            recall_values = [final_metrics[cat]['micro_metrics']['recall'] for cat in categories]
            f1_values = [final_metrics[cat]['micro_metrics']['f1'] for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.25
            
            plt.figure(figsize=(12, 7))
            bars1 = plt.bar(x - width, precision_values, width, label='Precision', color='#3498db')
            bars2 = plt.bar(x, recall_values, width, label='Recall', color='#2ecc71')
            bars3 = plt.bar(x + width, f1_values, width, label='F1', color='#e74c3c')
            
            # Add values on top of bars
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.ylabel('Score', fontsize=12)
            plt.title('Precision, Recall, and F1 per Category', fontsize=14)
            plt.xticks(x, categories)
            plt.ylim(0, 1.0)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300)
            plt.close()
        
        # Plot top and bottom performing classes for a specific category
        def plot_top_bottom_classes(category, n=5):
            metrics = final_metrics[category]['per_class']
            # Filter out None APs
            valid_metrics = {k: v for k, v in metrics.items() if v['AP'] is not None}
            
            # Sort by AP
            sorted_classes = sorted(valid_metrics.items(), key=lambda x: x[1]['AP'], reverse=True)
            
            top_classes = sorted_classes[:n]
            bottom_classes = sorted_classes[-n:]
            
            # Prepare data for top classes
            top_names = [item[0] for item in top_classes]
            top_aps = [item[1]['AP'] for item in top_classes]
            
            # Prepare data for bottom classes
            bottom_names = [item[0] for item in bottom_classes]
            bottom_aps = [item[1]['AP'] for item in bottom_classes]
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot top classes
            bars1 = ax1.barh(top_names, top_aps, color='#2ecc71')
            ax1.set_title(f'Top {n} {category.title()} by AP', fontsize=14)
            ax1.set_xlim(0, 1.0)
            ax1.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add values to bars
            for bar in bars1:
                width = bar.get_width()
                ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', ha='left', va='center', fontsize=10)
            
            # Plot bottom classes
            bars2 = ax2.barh(bottom_names, bottom_aps, color='#e74c3c')
            ax2.set_title(f'Bottom {n} {category.title()} by AP', fontsize=14)
            ax2.set_xlim(0, 1.0)
            ax2.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add values to bars
            for bar in bars2:
                width = bar.get_width()
                ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', ha='left', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{category}_top_bottom.png'), dpi=300)
            plt.close()
        
        # Create visualizations
        try:
            plot_map_comparison()
            plot_metrics_comparison()
            
            # Plot top/bottom classes for each category
            for category in ['instruments', 'verbs', 'iv_pairs']:
                plot_top_bottom_classes(category)
            
            print(f"\nVisualization saved to {output_dir}")
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    # Initialize ModelLoader
    try:
        loader = ModelLoader()
        
        # Load models
        yolo_model = loader.load_yolo_model()
        action_model = loader.load_action_model()
        
        # Dataset directory
        dataset_dir = str(loader.dataset_path)
        
        # Check dataset structure
        labels_dir = os.path.join(dataset_dir, "labels")
        videos_dir = os.path.join(dataset_dir, "videos")
        
        print("\nDataset Structure Check:")
        print(f"Labels Directory: {labels_dir}")
        print(f"Videos Directory: {videos_dir}")
        
        # Check available videos
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
        
        # Create evaluator
        evaluator = IntegratedEvaluator(
            yolo_model=yolo_model, 
            action_model=action_model, 
            dataset_dir=dataset_dir
        )
        
        # Perform evaluation
        results = evaluator.evaluate()
        
        # Save results and create visualizations
        output_dir = os.path.join(current_dir, "evaluation_results")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = np.datetime64('now').astype(str).replace(':', '-').replace(' ', '_')
        results_file = os.path.join(output_dir, f"eval_results_{timestamp}.json")
        vis_dir = os.path.join(output_dir, f"visualizations_{timestamp}")
        
        #evaluator.save_results_to_file(results, results_file)
        #evaluator.create_summary_visualizations(results, vis_dir)
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during initialization or evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()