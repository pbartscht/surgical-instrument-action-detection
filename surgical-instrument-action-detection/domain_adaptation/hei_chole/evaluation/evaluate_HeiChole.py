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
CONFIDENCE_THRESHOLD = 0.6
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

# Constants for Dataset Mappings
CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING = {
    'grasper': 'grasper',
    'bipolar': 'coagulation',
    'clipper': 'clipper',
    'hook': None,  # No direct mapping
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
        self.yolo_weights = self.hierarchical_dir / "Instrument-classification-detection" / "weights" / "instrument_detector" / "best_v35.pt"
        # Verb model path
        self.verb_model_path = self.hierarchical_dir / "verb_recognition/checkpoints/jumping-tree-47/last.ckpt"
        
        # Dataset path for HeiChole
        self.dataset_path = Path("/data/Bartscht/HeiChole")
        
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
        
        Args:
            instrument: Predicted instrument from CholecT50 model
            verb: Predicted verb from CholecT50 model
            
        Returns:
            Tuple (mapped_instrument, mapped_verb) in HeiChole format
        """
        mapped_instrument = CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING.get(instrument)
        mapped_verb = CHOLECT50_TO_HEICHOLE_VERB_MAPPING.get(verb)
        
        return mapped_instrument, mapped_verb

    def load_ground_truth(self, video):
        """
        Loads binary ground truth annotations for HeiChole dataset.
        
        Args:
            video: Video identifier (e.g., "VID01")
            
        Returns:
            Dictionary with binary frame annotations
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
                        # Convert to binary: 1 if present, 0 if not
                        frame_annotations[frame_number]['instruments'][instr_name] = 1 if present > 0 else 0
                    
                    # Process actions (binary)
                    actions = frame_data.get('actions', {})
                    for action_name, present in actions.items():
                        # Convert to binary: 1 if present, 0 if not
                        frame_annotations[frame_number]['actions'][action_name] = 1 if present > 0 else 0
                
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
            yolo_results = self.yolo_model(img)
            valid_detections = []
            
            # Process YOLO detections
            for detection in yolo_results[0].boxes:
                instrument_class = int(detection.cls)
                confidence = float(detection.conf)
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    cholect50_instrument = TOOL_MAPPING[instrument_class]
                    mapped_instrument, _ = self.map_cholect50_prediction(cholect50_instrument, None)
                    
                    if mapped_instrument:  # Only process if mapping exists
                        valid_detections.append({
                            'class': instrument_class,
                            'confidence': confidence,
                            'box': detection.xyxy[0],
                            'name': mapped_instrument
                        })
            
            # Sort by confidence
            valid_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Process each detection
            for detection in valid_detections:
                mapped_instrument = detection['name']
                box = detection['box']
                confidence = detection['confidence']
                
                # Get instrument crop for verb prediction
                x1, y1, x2, y2 = map(int, box)
                instrument_crop = img.crop((x1, y1, x2, y2))
                crop_tensor = self.transform(instrument_crop).unsqueeze(0).to(self.device)
                
                # Predict verb
                verb_outputs = self.verb_model(crop_tensor, [mapped_instrument])
                verb_probs = verb_outputs['probabilities']
                
                # Get top verb predictions
                top_verbs = []
                for verb_idx in torch.topk(verb_probs[0], k=3).indices.cpu().numpy():
                    cholect50_verb = VERB_MAPPING[VERB_MODEL_TO_EVAL_MAPPING[verb_idx]]
                    mapped_verb = CHOLECT50_TO_HEICHOLE_VERB_MAPPING.get(cholect50_verb)
                    
                    if mapped_verb is not None:
                        top_verbs.append({
                            'name': mapped_verb,
                            'probability': float(verb_probs[0][verb_idx])
                        })
                
                # Create prediction for best verb
                if top_verbs:
                    best_verb = max(top_verbs, key=lambda x: x['probability'])
                    verb_name = best_verb['name']
                    verb_conf = best_verb['probability']
                    
                    # Store binary prediction (confidence above threshold = 1)
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

    def evaluate(self, videos_to_analyze):
        """
        Evaluates model performance across specified videos using binary metrics.
        
        Args:
            videos_to_analyze: List of video IDs to evaluate
            
        Returns:
            Dictionary containing binary evaluation metrics
        """
        # Initialize binary prediction tracking
        predictions = {
            'instruments': defaultdict(list),  # {instrument_name: [{confidence, binary_gt}, ...]}
            'actions': defaultdict(list)      # {action_name: [{confidence, binary_gt}, ...]}
        }
        
        print("\nStarting binary evaluation process...")
        
        # Process each video
        for video in videos_to_analyze:
            print(f"\nProcessing {video}...")
            
            # Load ground truth
            ground_truth = self.load_ground_truth(video)
            
            # Get frame files
            video_folder = os.path.join(self.dataset_dir, "Videos", video)
            frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.png')])
            
            # Process frames
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
                
                # Update predictions and ground truth
                for pred in frame_predictions:
                    # Process instrument predictions
                    instrument = pred['instrument']
                    predictions['instruments'][instrument['name']].append({
                        'confidence': instrument['confidence'],
                        'binary_gt': gt_frame['instruments'].get(instrument['name'], 0)
                    })
                    
                    # Process action predictions
                    action = pred['action']
                    predictions['actions'][action['name']].append({
                        'confidence': action['confidence'],
                        'binary_gt': gt_frame['actions'].get(action['name'], 0)
                    })
        
        # Calculate binary metrics
        metrics = self._calculate_binary_metrics(predictions)
        
        return metrics

    def _calculate_binary_metrics(self, predictions):
        """
        Calculates binary classification metrics including AP.
        
        Args:
            predictions: Dictionary containing predictions and ground truth
            
        Returns:
            Dictionary with binary metrics
        """
        metrics = {
            'instruments': {},
            'actions': {}
        }
        
        # Calculate metrics for instruments and actions
        for category in ['instruments', 'actions']:
            category_metrics = {}
            
            for name, preds in predictions[category].items():
                if preds:  # Skip if no predictions
                    # Extract ground truth and confidence scores
                    y_true = np.array([p['binary_gt'] for p in preds])
                    y_scores = np.array([p['confidence'] for p in preds])
                    
                    # Calculate binary metrics
                    try:
                        # Average Precision (handles binary case automatically)
                        ap = average_precision_score(y_true, y_scores)
                        
                        # Store metrics
                        category_metrics[name] = {
                            'average_precision': ap,
                            'num_predictions': len(preds),
                            'num_positives': int(y_true.sum())
                        }
                    except Exception as e:
                        print(f"Error calculating metrics for {name}: {str(e)}")
                        continue
            
            # Calculate mean AP for category
            valid_aps = [m['average_precision'] for m in category_metrics.values() if m['average_precision'] is not None]
            mean_ap = np.mean(valid_aps) if valid_aps else 0.0
            
            metrics[category] = {
                'per_class': category_metrics,
                'mean_ap': mean_ap
            }
        
        return metrics
    
def analyze_label_distribution(dataset_dir, videos):
    """
    Analyzes the distribution of ground truth labels, including zero occurrences.
    
    Args:
        dataset_dir: Path to dataset directory
        videos: List of video IDs to analyze
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
    
    # Print detailed statistics
    print("\n==========================================")
    print("Ground Truth Label Distribution Analysis")
    print("==========================================")
    print(f"\nTotal frames analyzed: {total_frames}")
    
    # Print instrument frequencies
    print("\nINSTRUMENT FREQUENCIES:")
    print("=" * 50)
    print(f"{'Instrument':25s} {'Count':>8s} {'% of Frames':>12s} {'Present?':>10s}")
    print("-" * 50)
    
    for instr in sorted(all_possible_instruments):
        count = frequencies['instruments'][instr]
        percentage = (count / total_frames) * 100 if total_frames > 0 else 0
        present = "Yes" if count > 0 else "No"
        print(f"{instr:25s} {count:8d} {percentage:11.2f}% {present:>10s}")
    
    # Print action frequencies
    print("\nACTION FREQUENCIES:")
    print("=" * 50)
    print(f"{'Action':25s} {'Count':>8s} {'% of Frames':>12s} {'Present?':>10s}")
    print("-" * 50)
    
    for action in sorted(all_possible_actions):
        count = frequencies['actions'][action]
        percentage = (count / total_frames) * 100 if total_frames > 0 else 0
        present = "Yes" if count > 0 else "No"
        print(f"{action:25s} {count:8d} {percentage:11.2f}% {present:>10s}")
    
    return frequencies

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
        
        print("\n==========================================")
        print("GROUND TRUTH ANALYSIS")
        print("==========================================")
        
        # Ground Truth Analysis using analyze_label_distribution
        gt_distribution = analyze_label_distribution(dataset_dir, videos_to_analyze)
        
        # Running model predictions
        print("\n==========================================")
        print("MODEL PREDICTIONS ANALYSIS")
        print("==========================================")
        
        # Create evaluator and run predictions
        evaluator = HeiCholeEvaluator(
            yolo_model=yolo_model, 
            verb_model=verb_model, 
            dataset_dir=dataset_dir
        )
        
        results = evaluator.evaluate(videos_to_analyze)
        
        # Calculate total frames from all videos
        total_frames = sum(len(Path(dataset_dir) / "Videos" / video).glob("*.png") 
                         for video in videos_to_analyze)
        
        # Print final comparison
        print("\n==========================================")
        print("FINAL COMPARISON")
        print("==========================================")
        
        print("\nGROUND TRUTH DISTRIBUTION:")
        print("=" * 70)
        print(f"{'Category':20s} {'Count':>10s} {'Percentage':>12s}")
        print("-" * 70)
        
        print("\nINSTRUMENTS:")
        for instr, count in sorted(gt_distribution['instruments'].items()):
            percentage = (count / total_frames) * 100
            print(f"{instr:20s} {count:10d} {percentage:11.2f}%")
            
        print("\nACTIONS:")
        for action, count in sorted(gt_distribution['actions'].items()):
            percentage = (count / total_frames) * 100
            print(f"{action:20s} {count:10d} {percentage:11.2f}%")
        
        print("\nMODEL PREDICTIONS:")
        print("=" * 70)
        print(f"{'Category':20s} {'Predictions':>12s} {'AP Score':>10s}")
        print("-" * 70)
        
        print("\nINSTRUMENTS:")
        for instr, metrics in sorted(results['instruments']['per_class'].items()):
            pred_count = metrics['num_predictions']
            ap = metrics['average_precision']
            percentage = (pred_count / total_frames) * 100
            print(f"{instr:20s} {pred_count:10d} ({percentage:6.2f}%) {ap:8.4f}")
            
        print("\nACTIONS:")
        for action, metrics in sorted(results['actions']['per_class'].items()):
            pred_count = metrics['num_predictions']
            ap = metrics['average_precision']
            percentage = (pred_count / total_frames) * 100
            print(f"{action:20s} {pred_count:10d} ({percentage:6.2f}%) {ap:8.4f}")
        
        print("\nMEAN AP SCORES:")
        print("=" * 30)
        print(f"Instruments mAP: {results['instruments']['mean_ap']:.4f}")
        print(f"Actions mAP:     {results['actions']['mean_ap']:.4f}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()