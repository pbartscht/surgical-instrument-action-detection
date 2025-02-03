import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import json
from PIL import Image, ImageDraw
from collections import defaultdict
import numpy as np
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

# Constants
CONFIDENCE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.3

# Global mappings
TOOL_MAPPING = {
    0: 'grasper', 1: 'bipolar', 2: 'hook', 
    3: 'scissors', 4: 'clipper', 5: 'irrigator'
}

IGNORED_INSTRUMENTS = {
    6: 'specimen_bag'  # Index: Name der zu ignorierenden Instrumente
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

HEICHOLE_SPECIFIC_INSTRUMENTS = {
    'specimen_bag',
    'stapler'
}

class ModelLoader:
    def __init__(self):
        # Get the current script's directory and navigate to project root
        current_dir = Path(__file__).resolve().parent  # hei_chole/evaluation
        hei_chole_dir = current_dir.parent  # hei_chole
        domain_adaptation_dir = hei_chole_dir.parent  # domain_adaptation
        self.project_root = domain_adaptation_dir.parent  # surgical-instrument-action-detection
        self.hierarchical_dir = self.project_root / "models" / "hierarchical-surgical-workflow"
        self.setup_paths()

    def setup_paths(self):
        """Defines all important paths for the models"""
        # YOLO model path
        self.yolo_weights = self.hierarchical_dir / "Instrument-classification-detection" / "weights" / "instrument_detector" / "best_v35.pt"
        # Dataset path for HeiChole
        self.dataset_path = Path("/data/Bartscht/HeiChole/domain_adaptation/test")
        
        print(f"YOLO weights path: {self.yolo_weights}")
        print(f"Dataset path: {self.dataset_path}")

        # Validate paths
        if not self.yolo_weights.exists():
            raise FileNotFoundError(f"YOLO weights not found at: {self.yolo_weights}")
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.dataset_path}")

    def load_yolo_model(self):
        """Loads and returns the YOLO model"""
        try:
            model = YOLO(str(self.yolo_weights))
            print("YOLO model loaded successfully")
            return model
        except Exception as e:
            print(f"Error details: {str(e)}")
            raise Exception(f"Error loading YOLO model: {str(e)}")
        
class HeiCholeEvaluator:
    def __init__(self, yolo_model, dataset_dir):
        """
        Initialize the HeiChole evaluator.
        
        Args:
            yolo_model: Pre-trained YOLO model for instrument detection
            dataset_dir: Path to HeiChole dataset
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = yolo_model
        self.dataset_dir = dataset_dir

    def map_cholect50_prediction(self, instrument):
        """
        Maps CholecT50 predictions to HeiChole format.
        
        Args:
            instrument: Predicted instrument from CholecT50 model
            
        Returns:
            mapped_instrument in HeiChole format
        """
        return CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING.get(instrument)

    def load_ground_truth(self, video):
        """
        Loads ground truth annotations for HeiChole dataset.
        
        Args:
            video: Video identifier (e.g., "VID01")
            
        Returns:
            Dictionary with frame annotations
        """
        labels_folder = os.path.join(self.dataset_dir, "Labels")
        json_file = os.path.join(labels_folder, f"{video}.json")
        
        frame_annotations = defaultdict(lambda: {
            'instruments': defaultdict(int)
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
                        # Convert to binary: 1 if present, 0 if not
                        frame_annotations[frame_number]['instruments'][instr_name] = 1 if present > 0 else 0
                
                return frame_annotations
                
        except Exception as e:
            print(f"Error loading annotations: {str(e)}")
            raise

    def evaluate_frame(self, img_path, ground_truth, save_visualization=True):
        """
        Evaluates a single frame and maps predictions to HeiChole format.
        
        Args:
            img_path: Path to the frame image
            ground_truth: Ground truth annotations for this frame
            save_visualization: Whether to save visualization
            
        Returns:
            List of mapped predictions with binary confidence scores
        """
        frame_predictions = []
        frame_number = int(os.path.basename(img_path).split('.')[0])
        video_name = os.path.basename(os.path.dirname(img_path))
        
        img = Image.open(img_path)
        original_img = img.copy()
        draw = ImageDraw.Draw(original_img)
        
        try:
            # YOLO predictions
            import pdb; pdb.set_trace()  # Setzt einen Breakpoint
            yolo_results = self.yolo_model(img)
            valid_detections = []
            
            # Process YOLO detections
            for detection in yolo_results[0].boxes:
                instrument_class = int(detection.cls)
                confidence = float(detection.conf)
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Skip ignored instruments
                    if instrument_class in IGNORED_INSTRUMENTS:
                        print(f"\nFrame {frame_number}: Skipping ignored instrument {IGNORED_INSTRUMENTS[instrument_class]}")
                        continue
                    
                    # Get original CholecT50 instrument name
                    try:
                        cholect50_instrument = TOOL_MAPPING[instrument_class]
                    except KeyError:
                        print(f"\nWarning: Unknown instrument class {instrument_class}, skipping...")
                        continue
                    
                    # Map to HeiChole instrument
                    mapped_instrument = CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING.get(cholect50_instrument)
                    
                    print(f"\n{'='*50}")
                    print(f"Frame {frame_number} Detection:")
                    print(f"CholecT50 Instrument: {cholect50_instrument}")
                    print(f"HeiChole Instrument: {mapped_instrument}")
                    
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
                box = detection['box']
                confidence = detection['confidence']
                
                x1, y1, x2, y2 = map(int, box)
                
                prediction = {
                    'frame_id': f"{video_name}_frame{frame_number}",
                    'instrument': {
                        'name': mapped_instrument,
                        'confidence': confidence,
                        'binary_pred': 1 if confidence >= CONFIDENCE_THRESHOLD else 0
                    }
                }
                frame_predictions.append(prediction)
                
                # Visualization
                if save_visualization:
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                    text = f"{mapped_instrument}\nConf: {confidence:.2f}"
                    draw.text((x1, y1-20), text, fill='blue')
            
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
    def __init__(self, confidence_threshold=0.1):
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

    def calculate_metrics(self, predictions_per_frame, ground_truth):
        """
        Calculate binary classification metrics for instrument detection.
        
        Args:
            predictions_per_frame: Dictionary of predictions per frame
            ground_truth: Dictionary of ground truth annotations
            
        Returns:
            Dictionary containing per-class and mean metrics
        """
        results = {'per_class': {}, 'mean_metrics': {}}
        
        all_frame_numbers = sorted(list(ground_truth.keys()))
        num_frames = len(all_frame_numbers)
        frame_to_idx = {frame: idx for idx, frame in enumerate(all_frame_numbers)}
        
        # Three matrices: ground truth, binary predictions, and confidence scores
        y_true = np.zeros((num_frames, len(self.instrument_labels)), dtype=np.int32)
        y_pred = np.zeros((num_frames, len(self.instrument_labels)), dtype=np.int32)
        y_scores = np.zeros((num_frames, len(self.instrument_labels)), dtype=np.float32)
        
        # Fill ground truth matrix
        for frame_num, frame_data in ground_truth.items():
            frame_idx = frame_to_idx[frame_num]
            for label_idx, label in enumerate(self.instrument_labels):
                if label in frame_data['instruments']:
                    y_true[frame_idx, label_idx] = frame_data['instruments'][label]
        
        # Fill prediction and scores matrices
        for frame_id, preds in predictions_per_frame.items():
            frame_num = int(frame_id.split('_frame')[1])
            if frame_num in frame_to_idx:
                frame_idx = frame_to_idx[frame_num]
                for pred in preds:
                    name = pred['instrument']['name']
                    conf = pred['instrument']['confidence']
                        
                    try:
                        label_idx = self.instrument_labels.index(name)
                        y_scores[frame_idx, label_idx] = conf
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
        
        for i, label in enumerate(self.instrument_labels):
            # Calculate AP with confidence scores
            ap = average_precision_score(y_true[:, i], y_scores[:, i])
            
            results['per_class'][label] = {
                'f1_score': float(f1_scores[i]),
                'precision': float(precision_scores[i]),
                'recall': float(recall_scores[i]),
                'ap_score': float(ap),
                'support': int(ins_count_gt[i]),
                'predictions': int(ins_count_pred[i])
            }
            
            if ins_count_gt[i] > 0:  # Only count classes with ground truth
                overall_f1 += f1_scores[i]
                overall_ap += ap
                class_count += 1
        
        # Calculate mean metrics
        if class_count > 0:
            results['mean_metrics'] = {
                'mean_f1': overall_f1 / class_count,
                'mean_precision': np.mean(precision_scores[precision_scores > 0]),
                'mean_recall': np.mean(recall_scores[recall_scores > 0]),
                'mean_ap': overall_ap / class_count
            }
        
        return results

def analyze_label_distribution(dataset_dir, videos):
    """
    Analyzes the distribution of ground truth instrument labels.
    
    Args:
        dataset_dir: Path to dataset directory
        videos: List of video IDs to analyze
    """
    # Define all possible instruments
    all_possible_instruments = {
        'grasper', 'coagulation', 'clipper', 'scissors', 
        'suction_irrigation', 'specimen_bag', 'stapler'
    }
    
    # Initialize counters
    frequencies = defaultdict(int)
    total_frames = 0
    
    print("\nAnalyzing ground truth instrument distribution...")
    
    # Process each video
    for video in videos:
        print(f"\nProcessing {video}...")
        json_file = os.path.join(dataset_dir, "Labels", f"{video}.json")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                frames = data.get('frames', {})
                total_frames += len(frames)
                
                # Count instrument occurrences
                for frame_data in frames.values():
                    instruments = frame_data.get('instruments', {})
                    for instr, present in instruments.items():
                        if present > 0:
                            frequencies[instr] += 1
        
        except Exception as e:
            print(f"Error processing {video}: {str(e)}")
            continue
    
    # Print detailed statistics
    print("\n==========================================")
    print("Ground Truth Instrument Distribution Analysis")
    print("==========================================")
    print(f"\nTotal frames analyzed: {total_frames}")
    
    # Print instrument frequencies
    print("\nINSTRUMENT FREQUENCIES:")
    print("=" * 50)
    print(f"{'Instrument':25s} {'Count':>8s} {'% of Frames':>12s} {'Present?':>10s}")
    print("-" * 50)
    
    for instr in sorted(all_possible_instruments):
        count = frequencies[instr]
        percentage = (count / total_frames) * 100 if total_frames > 0 else 0
        present = "Yes" if count > 0 else "No"
        print(f"{instr:25s} {count:8d} {percentage:11.2f}% {present:>10s}")
    
    return frequencies

def print_metrics_report(metrics, total_frames):
    """
    Prints a formatted report of the metrics.
    
    Args:
        metrics: Dictionary containing calculated metrics
        total_frames: Total number of frames analyzed
    """
    print("\n====== INSTRUMENT DETECTION EVALUATION REPORT ======")
    print("=" * 70)
    print(f"{'Instrument':15s} {'F1-Score':>10s} {'Precision':>10s} {'Recall':>10s} {'AP':>10s} {'Support':>10s} {'Predictions':>12s}")
    print("-" * 70)
    
    for label, scores in metrics['per_class'].items():
        pred_count = scores['predictions']
        percentage = (pred_count / total_frames) * 100 if total_frames > 0 else 0
        
        print(f"{label:15s} {scores['f1_score']:10.4f} {scores['precision']:10.4f} "
              f"{scores['recall']:10.4f} {scores['ap_score']:10.4f} {scores['support']:10d} "
              f"{pred_count:8d} ({percentage:5.1f}%)")
    
    print("\nMEAN METRICS:")
    print("-" * 30)
    means = metrics['mean_metrics']
    print(f"Mean AP:        {means['mean_ap']:.4f}")
    print(f"Mean F1-Score:  {means['mean_f1']:.4f}")
    print(f"Mean Precision: {means['mean_precision']:.4f}")
    print(f"Mean Recall:    {means['mean_recall']:.4f}")

def main():
    """Compare ground truth and model predictions for selected videos in HeiChole dataset"""
    try:
        # Initialize ModelLoader
        loader = ModelLoader()
        
        # Load YOLO model only
        yolo_model = loader.load_yolo_model()
        dataset_dir = str(loader.dataset_path)
        
        # Specify videos to analyze
        videos_to_analyze = ["VID08", "VID13"]
        print(f"\nAnalyzing videos: {', '.join(videos_to_analyze)}")
        
        print("\n==========================================")
        print("GROUND TRUTH ANALYSIS")
        print("==========================================")
        
        # Ground Truth Analysis
        gt_distribution = analyze_label_distribution(dataset_dir, videos_to_analyze)
        
        print("\n==========================================")
        print("MODEL PREDICTIONS ANALYSIS")
        print("==========================================")
        
        # Create evaluator (YOLO only)
        evaluator = HeiCholeEvaluator(
            yolo_model=yolo_model,
            dataset_dir=dataset_dir
        )
        
        # Initialize Binary Metrics Calculator
        metrics_calculator = BinaryMetricsCalculator(confidence_threshold=CONFIDENCE_THRESHOLD)
        
        # Collect Predictions and Ground Truth
        predictions_per_frame = {}
        ground_truth = {}
        total_frames = 0
        
        # Process each video
        for video in videos_to_analyze:
            # Load Ground Truth
            gt = evaluator.load_ground_truth(video)
            ground_truth.update(gt)
            
            # Process each frame
            video_folder = os.path.join(dataset_dir, "Videos", video)
            for frame_file in os.listdir(video_folder):
                if frame_file.endswith('.png'):
                    total_frames += 1
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
        
        print(f"\nProcessed {total_frames} frames in total.")
        
        # Calculate metrics
        binary_metrics = metrics_calculator.calculate_metrics(predictions_per_frame, ground_truth)
        
        # Print detailed metrics report
        print_metrics_report(binary_metrics, total_frames)
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()