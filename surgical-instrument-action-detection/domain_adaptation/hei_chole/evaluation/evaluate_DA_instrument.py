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
sys.path.append(str(hei_chole_dir))
from experiments import FeatureAlignmentHead, InstrumentDetector

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

# Constants and Mappings
CONFIDENCE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.3

TOOL_MAPPING = {
    0: 'grasper', 1: 'bipolar', 2: 'hook', 
    3: 'scissors', 4: 'clipper', 5: 'irrigator'
}

IGNORED_INSTRUMENTS = {
    6: 'specimen_bag'
}

VERB_MAPPING = {
    0: 'grasp', 1: 'retract', 2: 'dissect', 3: 'coagulate', 
    4: 'clip', 5: 'cut', 6: 'aspirate', 7: 'irrigate', 
    8: 'pack', 9: 'null_verb'
}

CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING = {
    'grasper': 'grasper',
    'bipolar': 'coagulation',
    'clipper': 'clipper',
    'hook': 'coagulation',
    'scissors': 'scissors',
    'irrigator': 'suction_irrigation'
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
        self.yolo_weights = (self.hierarchical_dir / "Instrument-classification-detection" / 
                           "weights" / "instrument_detector" / "best_v35.pt")
        self.alignment_head_weights = (self.project_root / "domain_adaptation" / 
                                     "hei_chole" / "experiments" / "checkpoints" / 
                                     "alignment_head_epoch_46.pt")
        self.dataset_path = Path("/data/Bartscht/HeiChole")
        
        print(f"\nPath Debug Info:")
        print(f"Project Root: {self.project_root}")
        print(f"Hierarchical Dir: {self.hierarchical_dir}")
        print(f"YOLO weights path: {self.yolo_weights}")
        print(f"Alignment head path: {self.alignment_head_weights}")
        print(f"Dataset path: {self.dataset_path}")
        
        # Validate paths
        if not self.yolo_weights.exists():
            raise FileNotFoundError(f"YOLO weights not found at: {self.yolo_weights}")
        if not self.alignment_head_weights.exists():
            raise FileNotFoundError(f"Alignment head weights not found at: {self.alignment_head_weights}")
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.dataset_path}")

    def load_yolo_model(self):
        try:
            model = YOLO(str(self.yolo_weights))
            alignment_head = FeatureAlignmentHead(num_classes=len(TOOL_MAPPING))
            checkpoint = torch.load(str(self.alignment_head_weights), map_location='cuda')
            alignment_head.load_state_dict(checkpoint['model_state_dict'])
            alignment_head.eval()
            detector = InstrumentDetector(model, alignment_head)
            return detector
        except Exception as e:
            raise Exception(f"Error loading models: {str(e)}")

class InstrumentDetector:
    def __init__(self, yolo_model, alignment_head):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = yolo_model.to(self.device)
        self.alignment_head = alignment_head.to(self.device)
        
    def extract_features(self, img):
        """Extract features from YOLO's backbone"""
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
            
        x = img.to(self.device)
        features = None
        
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model.model.model):
                x = layer(x)
                if i == 10:  # C2PSA layer
                    features = x
                    break
        return features
    
    def __call__(self, img):
        """Process image through both YOLO and Alignment Head"""
        img = img.to(self.device)
        
        # Extract features
        features = self.extract_features(img)
        
        # Get YOLO predictions
        yolo_results = self.yolo_model(img)
        
        # Get Alignment Head predictions
        with torch.no_grad():
            _, refined_preds = self.alignment_head(features)
            
        return {
            'yolo_results': yolo_results,
            'alignment_preds': refined_preds,
            'features': features
        }

def create_prediction(frame_id, instrument_name, confidence, action_name=None, action_conf=None):
    """Helper function to create standardized prediction dictionary"""
    prediction = {
        'frame_id': frame_id,
        'instrument': {
            'name': CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING.get(instrument_name),
            'confidence': confidence,
            'binary_pred': 1 if confidence >= CONFIDENCE_THRESHOLD else 0
        }
    }
    
    if action_name and action_conf is not None:
        prediction['action'] = {
            'name': action_name,
            'confidence': action_conf,
            'binary_pred': 1 if action_conf >= CONFIDENCE_THRESHOLD else 0
        }
    
    return prediction

class HeiCholeEvaluator:
    def __init__(self, yolo_model, dataset_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = yolo_model
        self.dataset_dir = dataset_dir
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_ground_truth(self, video):
        """Loads ground truth annotations"""
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
                    
                    instruments = frame_data.get('instruments', {})
                    for instr_name, present in instruments.items():
                        frame_annotations[frame_number]['instruments'][instr_name] = 1 if present > 0 else 0
                    
                    actions = frame_data.get('actions', {})
                    for action_name, present in actions.items():
                        frame_annotations[frame_number]['actions'][action_name] = 1 if present > 0 else 0
                
                return frame_annotations
                
        except Exception as e:
            raise Exception(f"Error loading annotations: {str(e)}")

    def evaluate_frame(self, img_path, ground_truth):
        """Evaluates a single frame with enhanced Alignment Head integration"""
        frame_predictions = {
            'yolo_only': [],
            'alignment_head': [],
            'combined': []
        }
        
        frame_number = int(os.path.basename(img_path).split('.')[0])
        video_name = os.path.basename(os.path.dirname(img_path))
        
        try:
            # Load and preprocess image
            img = Image.open(img_path)
            img_tensor = transforms.ToTensor()(img)
            img_tensor = img_tensor.unsqueeze(0)
            
            # Get predictions from both models
            results = self.yolo_model(img_tensor)
            
            # Process YOLO predictions
            for box_info in results['yolo_results'][0].boxes:
                cls_idx = int(box_info.cls)
                if cls_idx in IGNORED_INSTRUMENTS:
                    continue
                    
                yolo_conf = float(box_info.conf)
                instrument_name = TOOL_MAPPING[cls_idx]
                
                yolo_pred = create_prediction(
                    frame_id=f"{video_name}_frame{frame_number}",
                    instrument_name=instrument_name,
                    confidence=yolo_conf
                )
                frame_predictions['yolo_only'].append(yolo_pred)
            
            # Process Alignment Head predictions with direct binary classification
            align_preds = results['alignment_preds']
            binary_preds = (align_preds > 0.5).float()
            
            for cls_idx, (is_present, conf) in enumerate(zip(binary_preds[0], align_preds[0])):
                if is_present:  # Use direct binary prediction
                    instrument_name = TOOL_MAPPING[cls_idx]
                    align_pred = create_prediction(
                        frame_id=f"{video_name}_frame{frame_number}",
                        instrument_name=instrument_name,
                        confidence=float(conf)
                    )
                    frame_predictions['alignment_head'].append(align_pred)
                    
                    if float(conf) > 0.8:  # Debug high confidence predictions
                        print(f"High confidence alignment prediction: {instrument_name} ({conf:.4f})")
            
            # Combine predictions with enhanced Alignment Head trust
            frame_predictions['combined'] = self.combine_predictions(
                frame_predictions['yolo_only'],
                frame_predictions['alignment_head'],
                trust_alignment_head=True
            )
            
            return frame_predictions
            
        except Exception as e:
            print(f"Error processing frame {frame_number}: {str(e)}")
            raise

    def combine_predictions(self, yolo_preds, align_preds, trust_alignment_head=True):
        """Enhanced prediction combination with stronger Alignment Head influence"""
        combined = []
        
        yolo_map = {pred['instrument']['name']: pred for pred in yolo_preds}
        align_map = {pred['instrument']['name']: pred for pred in align_preds}
        
        all_instruments = set(list(yolo_map.keys()) + list(align_map.keys()))
        
        for instrument in all_instruments:
            yolo_pred = yolo_map.get(instrument)
            align_pred = align_map.get(instrument)
            
            if align_pred and trust_alignment_head:
                # Trust Alignment Head prediction when available
                combined_conf = float(align_pred['instrument']['confidence'])
                binary_pred = align_pred['instrument']['binary_pred']
            elif yolo_pred and align_pred:
                # Weighted combination favoring Alignment Head
                combined_conf = (0.3 * float(yolo_pred['instrument']['confidence']) + 
                               0.7 * float(align_pred['instrument']['confidence']))
                binary_pred = 1 if combined_conf >= self.confidence_threshold else 0
            elif yolo_pred:
                # Fallback to YOLO
                combined_conf = float(yolo_pred['instrument']['confidence'])
                binary_pred = yolo_pred['instrument']['binary_pred']
            else:
                continue  # Skip if no valid prediction
            
            combined.append({
                'instrument': {
                    'name': instrument,
                    'confidence': combined_conf,
                    'binary_pred': binary_pred
                }
            })
        
        return combined
    
class EnhancedMetricsCalculator:
    def __init__(self, confidence_threshold=0.1):
        self.confidence_threshold = confidence_threshold
        
        # Define fixed order of labels
        self.instrument_labels = [
            'grasper', 'clipper', 'coagulation',
            'scissors', 'suction_irrigation',
            'specimen_bag', 'stapler'
        ]
        
        self.action_labels = [
            'grasp', 'hold', 'cut', 'clip'
        ]

    def calculate_all_metrics(self, predictions_per_frame, ground_truth):
        """
        Berechnet umfassende Metriken für alle drei Prediktionsarten
        """
        results = {
            'yolo_only': self._calculate_model_metrics(predictions_per_frame, ground_truth, 'yolo_only'),
            'alignment_head': self._calculate_model_metrics(predictions_per_frame, ground_truth, 'alignment_head'),
            'combined': self._calculate_model_metrics(predictions_per_frame, ground_truth, 'combined')
        }
        
        return results

    def _calculate_model_metrics(self, predictions_per_frame, ground_truth, model_type):
        """
        Berechnet detaillierte Metriken für einen spezifischen Modelltyp
        """
        results = {
            'instruments': {'per_class': {}, 'mean_metrics': {}},
            'actions': {'per_class': {}, 'mean_metrics': {}}
        }
        
        # Get all frame numbers and create frame index mapping
        all_frame_numbers = sorted(list(ground_truth.keys()))
        num_frames = len(all_frame_numbers)
        frame_to_idx = {frame: idx for idx, frame in enumerate(all_frame_numbers)}
        
        for category, label_list in [('instruments', self.instrument_labels), 
                                   ('actions', self.action_labels)]:
                                       
            # Initialize matrices for ground truth, predictions and confidence scores
            y_true = np.zeros((num_frames, len(label_list)), dtype=np.int32)
            y_pred = np.zeros((num_frames, len(label_list)), dtype=np.int32)
            y_scores = np.zeros((num_frames, len(label_list)), dtype=np.float32)
            
            # Fill ground truth matrix
            for frame_num, frame_data in ground_truth.items():
                frame_idx = frame_to_idx[frame_num]
                for label_idx, label in enumerate(label_list):
                    if label in frame_data[category]:
                        y_true[frame_idx, label_idx] = frame_data[category][label]
            
            # Fill prediction and confidence matrices
            for frame_id, preds in predictions_per_frame.items():
                if model_type not in preds:
                    continue
                    
                frame_num = int(frame_id.split('_frame')[1])
                if frame_num in frame_to_idx:
                    frame_idx = frame_to_idx[frame_num]
                    for pred in preds[model_type]:
                        if category == 'instruments':
                            name = pred['instrument']['name']
                            conf = pred['instrument']['confidence']
                        else:
                            if 'action' not in pred:
                                continue
                            name = pred['action']['name']
                            conf = pred['action']['confidence']
                            
                        try:
                            label_idx = label_list.index(name)
                            y_scores[frame_idx, label_idx] = conf
                            y_pred[frame_idx, label_idx] = 1 if conf >= self.confidence_threshold else 0
                        except ValueError:
                            continue
            
            # Calculate metrics for each class
            results[category] = self._calculate_class_metrics(y_true, y_pred, y_scores, label_list)
            
        return results

    def _calculate_class_metrics(self, y_true, y_pred, y_scores, label_list):
        """
        Berechnet detaillierte Metriken für jede Klasse
        """
        metrics = {
            'per_class': {},
            'mean_metrics': {}
        }
        
        # Calculate per-class metrics
        f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
        precision_scores = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_scores = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        # Count instances
        pred_counts = np.sum(y_pred, axis=0)
        gt_counts = np.sum(y_true, axis=0)
        
        # Calculate metrics for each class
        overall_f1 = 0
        overall_ap = 0
        class_count = 0
        
        for i, label in enumerate(label_list):
            ap = average_precision_score(y_true[:, i], y_scores[:, i])
            
            metrics['per_class'][label] = {
                'f1_score': float(f1_scores[i]),
                'precision': float(precision_scores[i]),
                'recall': float(recall_scores[i]),
                'ap_score': float(ap),
                'support': int(gt_counts[i]),
                'predictions': int(pred_counts[i])
            }
            
            if gt_counts[i] > 0:
                overall_f1 += f1_scores[i]
                overall_ap += ap
                class_count += 1
        
        # Calculate mean metrics
        if class_count > 0:
            metrics['mean_metrics'] = {
                'mean_f1': overall_f1 / class_count,
                'mean_precision': np.mean(precision_scores[precision_scores > 0]),
                'mean_recall': np.mean(recall_scores[recall_scores > 0]),
                'mean_ap': overall_ap / class_count
            }
            
        return metrics

def print_comparative_metrics(metrics_dict):
    """
    Druckt einen detaillierten Vergleich der Metriken aller Modelle
    mit korrigierter Formatierung
    """
    print("\n====== COMPARATIVE EVALUATION REPORT ======")
    
    models = ['yolo_only', 'alignment_head', 'combined']
    categories = ['instruments', 'actions']
    
    for category in categories:
        print(f"\n{category.upper()} PERFORMANCE COMPARISON:")
        print("=" * 100)
        
        # Header
        print(f"{'Model':<20} {'mAP':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'Support':>10}")
        print("-" * 100)
        
        # Model metrics
        for model in models:
            metrics = metrics_dict[model][category]['mean_metrics']
            print(f"{model:<20} "
                  f"{metrics['mean_ap']:10.4f} "
                  f"{metrics['mean_f1']:10.4f} "
                  f"{metrics['mean_precision']:10.4f} "
                  f"{metrics['mean_recall']:10.4f}")
        
        # Per-class breakdown
        print("\nPER-CLASS BREAKDOWN:")
        print("=" * 100)
        
        # Header for per-class metrics
        print(f"{'Class':<15} {'Model':<15} {'AP':>10} {'F1':>10} {'Precision':>10} "
              f"{'Recall':>10} {'Support':>10}")
        print("-" * 100)
        
        # Print per-class metrics
        classes = metrics_dict['yolo_only'][category]['per_class'].keys()
        for cls in classes:
            for i, model in enumerate(models):
                metrics = metrics_dict[model][category]['per_class'][cls]
                
                # Only print class name for first model
                class_name = cls if i == 0 else ""
                
                print(f"{class_name:<15} "
                      f"{model:<15} "
                      f"{metrics['ap_score']:10.4f} "
                      f"{metrics['f1_score']:10.4f} "
                      f"{metrics['precision']:10.4f} "
                      f"{metrics['recall']:10.4f} "
                      f"{metrics['support']:10d}")
            print("-" * 100)

def analyze_label_distribution(dataset_dir, videos):
    """
    Analysiert die Verteilung der Ground Truth Labels
    """
    all_possible_instruments = {
        'grasper', 'coagulation', 'clipper', 'scissors', 
        'suction_irrigation', 'specimen_bag', 'stapler'
    }
    
    all_possible_actions = {
        'grasp', 'hold', 'clip', 'cut'
    }
    
    frequencies = {
        'instruments': defaultdict(int),
        'actions': defaultdict(int)
    }
    
    total_frames = 0
    
    print("\nAnalyzing ground truth label distribution...")
    
    for video in videos:
        print(f"\nProcessing {video}...")
        json_file = os.path.join(dataset_dir, "Labels", f"{video}.json")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                frames = data.get('frames', {})
                total_frames += len(frames)
                
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
    
    print_distribution_statistics(frequencies, total_frames)
    return frequencies

def print_distribution_statistics(frequencies, total_frames):
    """
    Druckt detaillierte Statistiken über die Label-Verteilung
    """
    print("\n==========================================")
    print("Ground Truth Label Distribution Analysis")
    print("==========================================")
    print(f"\nTotal frames analyzed: {total_frames}")
    
    for category in ['instruments', 'actions']:
        print(f"\n{category.upper()} FREQUENCIES:")
        print("=" * 60)
        print(f"{'Label':25s} {'Count':>8s} {'% of Frames':>12s} {'Present?':>10s}")
        print("-" * 60)
        
        for label in sorted(frequencies[category].keys()):
            count = frequencies[category][label]
            percentage = (count / total_frames) * 100 if total_frames > 0 else 0
            present = "Yes" if count > 0 else "No"
            print(f"{label:25s} {count:8d} {percentage:11.2f}% {present:>10s}")

def main():
    """
    Hauptfunktion für die Evaluierung der Modelle
    """
    try:
        # Initialize ModelLoader
        loader = ModelLoader()
        
        # Load models
        print("\nLoading models...")
        detector = loader.load_yolo_model()
        dataset_dir = str(loader.dataset_path)
        
        # Get videos to analyze
        videos_to_analyze = ["VID13"]
        print(f"\nAnalyzing video: {videos_to_analyze[0]}")
        
        # Analyze ground truth distribution
        print("\nAnalyzing ground truth distribution...")
        gt_distribution = analyze_label_distribution(dataset_dir, videos_to_analyze)
        
        # Initialize evaluator and metrics calculator
        print("\nInitializing evaluator...")
        evaluator = HeiCholeEvaluator(
            yolo_model=detector,
            dataset_dir=dataset_dir
        )
        
        metrics_calculator = EnhancedMetricsCalculator(confidence_threshold=CONFIDENCE_THRESHOLD)
        
        # Collect predictions and ground truth
        print("\nCollecting predictions and ground truth...")
        predictions_per_frame = {}
        ground_truth = {}
        
        # Process each video
        for video in tqdm(videos_to_analyze, desc="Processing videos"):
            # Load ground truth
            gt = evaluator.load_ground_truth(video)
            ground_truth.update(gt)
            
            # Process each frame
            video_folder = os.path.join(dataset_dir, "Videos", video)
            for frame_file in tqdm(os.listdir(video_folder), desc=f"Processing frames in {video}"):
                if frame_file.endswith('.png'):
                    frame_id = f"{video}_frame{frame_file.split('.')[0]}"
                    img_path = os.path.join(video_folder, frame_file)
                    
                    # Get predictions from all models
                    frame_predictions = evaluator.evaluate_frame(
                        img_path,
                        gt[int(frame_file.split('.')[0])],
                    )
                    
                    predictions_per_frame[frame_id] = frame_predictions
        
        # Calculate metrics
        print("\nCalculating metrics...")
        all_metrics = metrics_calculator.calculate_all_metrics(
            predictions_per_frame, ground_truth
        )
        
        # Print results
        print("\nGenerating evaluation report...")
        print_comparative_metrics(all_metrics)
        
        # Save results to file
        results_file = os.path.join(dataset_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()