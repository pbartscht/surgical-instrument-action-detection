import os
import sys
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import ultralytics.nn.modules.conv
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import traceback

# Evaluation Constants
CONFIDENCE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.3

# Global Instrument Mappings
TOOL_MAPPING = {
    0: 'grasper', 1: 'bipolar', 2: 'hook', 
    3: 'scissors', 4: 'clipper', 5: 'irrigator'
}

IGNORED_INSTRUMENTS = {
    6: 'specimen_bag'  # Index: Name of instruments to ignore
}

# Dataset Mapping Constants
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

# Fixed order of labels matching ground truth JSON structure
INSTRUMENT_LABELS = [
    'grasper',
    'clipper', 
    'coagulation',
    'scissors',
    'suction_irrigation',
    'specimen_bag',
    'stapler'
]

class ModelLoader:
    def __init__(self):
        # Get the current script's directory and navigate to project root
        current_dir = Path(__file__).resolve().parent
        hei_chole_dir = current_dir.parent
        domain_adaptation_dir = hei_chole_dir.parent
        self.project_root = domain_adaptation_dir.parent
        self.hierarchical_dir = self.project_root / "models" / "hierarchical-surgical-workflow"
        self.setup_paths()

    def setup_paths(self):
        """Defines all important paths for the models"""
        self.yolo_weights = self.hierarchical_dir / "Instrument-classification-detection" / "weights" / "instrument_detector" / "best_v35.pt"
        self.dataset_path = Path("/data/Bartscht/HeiChole/domain_adaptation/test")
        
        print(f"YOLO weights path: {self.yolo_weights}")
        print(f"Dataset path: {self.dataset_path}")

        if not self.yolo_weights.exists():
            raise FileNotFoundError(f"YOLO weights not found at: {self.yolo_weights}")
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.dataset_path}")

def load_yolo_model(weights_path):
    """Loads YOLO model in inference mode only"""
    try:
        model = YOLO(str(weights_path))
        # Force inference mode
        model.model.eval()
        # This next line is crucial - it forces YOLO into pure inference mode
        model.predict(source=None, stream=True)
        
        # Double check everything is in eval mode
        model.model.eval()
        for param in model.model.parameters():
            param.requires_grad = False
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {str(e)}")
        raise

class BinaryMetricsCalculator:
    def __init__(self, confidence_threshold=0.1):
        self.confidence_threshold = confidence_threshold
        self.instrument_labels = INSTRUMENT_LABELS

    def calculate_metrics(self, predictions_per_frame, ground_truth):
        """Calculate binary classification metrics for instrument detection"""
        results = {'per_class': {}, 'mean_metrics': {}}
        
        all_frame_numbers = sorted(list(ground_truth.keys()))
        num_frames = len(all_frame_numbers)
        frame_to_idx = {frame: idx for idx, frame in enumerate(all_frame_numbers)}
        
        # Initialize matrices
        y_true = np.zeros((num_frames, len(self.instrument_labels)), dtype=np.int32)
        y_pred = np.zeros((num_frames, len(self.instrument_labels)), dtype=np.int32)
        y_scores = np.zeros((num_frames, len(self.instrument_labels)), dtype=np.float32)
        
        # Fill matrices
        self._fill_ground_truth_matrix(y_true, ground_truth, frame_to_idx)
        self._fill_prediction_matrices(y_pred, y_scores, predictions_per_frame, frame_to_idx)
        
        # Calculate metrics
        return self._calculate_all_metrics(y_true, y_pred, y_scores)

    def _fill_ground_truth_matrix(self, y_true, ground_truth, frame_to_idx):
        for frame_num, frame_data in ground_truth.items():
            frame_idx = frame_to_idx[frame_num]
            for label_idx, label in enumerate(self.instrument_labels):
                if label in frame_data['instruments']:
                    y_true[frame_idx, label_idx] = frame_data['instruments'][label]

    def _fill_prediction_matrices(self, y_pred, y_scores, predictions_per_frame, frame_to_idx):
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

    def _calculate_all_metrics(self, y_true, y_pred, y_scores):
        results = {'per_class': {}, 'mean_metrics': {}}
        
        # Calculate base metrics
        f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
        precision_scores = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_scores = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        # Count instances
        ins_count_pred = np.sum(y_pred, axis=0)
        ins_count_gt = np.sum(y_true, axis=0)
        
        # Calculate per-class metrics
        overall_f1 = 0
        overall_ap = 0
        class_count = 0
        
        for i, label in enumerate(self.instrument_labels):
            ap = average_precision_score(y_true[:, i], y_scores[:, i])
            
            results['per_class'][label] = {
                'f1_score': float(f1_scores[i]),
                'precision': float(precision_scores[i]),
                'recall': float(recall_scores[i]),
                'ap_score': float(ap),
                'support': int(ins_count_gt[i]),
                'predictions': int(ins_count_pred[i])
            }
            
            if ins_count_gt[i] > 0:
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
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class DomainAdapter(nn.Module):
    def __init__(self, yolo_path):
        super().__init__()
        model = YOLO(yolo_path)
        model.model.eval()
        model.predict(source=None, stream=True)
        self.yolo_model = model.model.model
        self.feature_layer = 9
        
        for param in self.yolo_model.parameters():
            param.requires_grad = False
            
        
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=32),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, alpha=1.0, return_features=False):
        with torch.no_grad():
            features = None
            for i, layer in enumerate(self.yolo_model):
                if i > self.feature_layer:
                    break
                x = layer(x)
                if i == self.feature_layer:
                    features = x.clone()
            
            reduced_features = self.feature_reducer(features)
            if return_features:
                return None, reduced_features
            return None


class DomainAdaptedYOLO(nn.Module):
    def __init__(self, yolo_path, feature_reducer):
        super().__init__()
        
        # Load YOLO model in inference mode
        model = YOLO(yolo_path)
        model.model.eval()  # Set to evaluation mode
        model.predict(source=None, stream=True)  # Initialize in inference mode
        self.yolo_model = model.model
        
        self.feature_reducer = feature_reducer
        self.feature_layer = 9  # SPPF layer
        
        # Register hook for feature modification
        for i, layer in enumerate(self.yolo_model.model):
            if i == self.feature_layer:
                layer.register_forward_hook(self._feature_hook)
        
        # Ensure everything is in eval mode
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
            
    def _feature_hook(self, module, input_feat, output_feat):
        """Hook für Feature Modification nach Layer 9"""
        try:
            print(f"Input shape: {[f.shape for f in input_feat]}")
            print(f"Output shape vor Feature Reducer: {output_feat.shape}")
            
            # Apply feature reducer
            modified = self.feature_reducer(output_feat)
            print(f"Output shape nach Feature Reducer: {modified.shape}")
            
            return modified
            
        except Exception as e:
            print(f"Error in hook: {str(e)}")
            return output_feat
    
    def forward(self, x):
        """
        Forward pass - strict inference mode
        """
        with torch.no_grad():
            try:
                # Ensure input tensor format
                if not isinstance(x, torch.Tensor):
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])
                    ])
                    x = transform(x).unsqueeze(0)
                
                if x.device != next(self.parameters()).device:
                    x = x.to(next(self.parameters()).device)
                
                # Process through model
                output = self.yolo_model(x)
                
                # Convert predictions to interpretable format
                detect_layer = self.yolo_model.model[-1]  # Get Detect layer
                if isinstance(detect_layer, ultralytics.nn.modules.head.Detect):
                    predictions = []
                    
                    # Process each detection
                    for i, pred in enumerate(output):
                        boxes = pred.boxes  # Get boxes for this prediction
                        for box in boxes:
                            prediction = {
                                'box': box.xyxy[0].cpu().numpy(),  # Convert box to numpy
                                'confidence': float(box.conf),  # Confidence score
                                'class': int(box.cls),  # Class index
                                'class_name': TOOL_MAPPING.get(int(box.cls), 'unknown')  # Map to class name
                            }
                            predictions.append(prediction)
                    
                    print(f"Found {len(predictions)} detections")
                    for pred in predictions:
                        print(f"Detected {pred['class_name']} with confidence {pred['confidence']:.2f}")
                    
                    return predictions
                else:
                    print("Warning: Last layer is not Detect layer")
                    return output
                
            except Exception as e:
                print(f"Error in forward pass: {str(e)}")
                traceback.print_exc()
                raise

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
        """Maps CholecT50 predictions to HeiChole format"""
        return CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING.get(instrument)

    def load_ground_truth(self, video):
        """Loads ground truth annotations for HeiChole dataset"""
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
                    instruments = frame_data.get('instruments', {})
                    for instr_name, present in instruments.items():
                        # Convert to binary: 1 if present, 0 if not
                        frame_annotations[frame_number]['instruments'][instr_name] = 1 if present > 0 else 0
                
                return frame_annotations
                
        except Exception as e:
            print(f"Error loading annotations: {str(e)}")
            raise

    def evaluate_frame(self, img_path, ground_truth, save_visualization=True):
        """Evaluates a single frame and maps predictions to HeiChole format"""
        frame_predictions = []
        frame_number = int(os.path.basename(img_path).split('.')[0])
        video_name = os.path.basename(os.path.dirname(img_path))
        
        img = Image.open(img_path)
        original_img = img.copy()
        draw = ImageDraw.Draw(original_img)
        
        try:
            # Get YOLO predictions
            with torch.no_grad():
                yolo_results = self.yolo_model(img)
                valid_detections = []
                
                # Process YOLO detections
                for detection in yolo_results[0].boxes:
                    instrument_class = int(detection.cls)
                    confidence = float(detection.conf)
                    
                    if confidence >= CONFIDENCE_THRESHOLD:
                        # Skip ignored instruments
                        if instrument_class in IGNORED_INSTRUMENTS:
                            continue
                        
                        # Get original CholecT50 instrument name
                        try:
                            cholect50_instrument = TOOL_MAPPING[instrument_class]
                        except KeyError:
                            continue
                        
                        # Map to HeiChole instrument
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
        
class EnhancedHeiCholeEvaluator(HeiCholeEvaluator):
    def __init__(self, yolo_model, domain_adapter, dataset_dir):
        """
        Initialize the evaluator with both base YOLO and domain-adapted YOLO.
        """
        super().__init__(yolo_model, dataset_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ensure domain adapter is in eval mode
        domain_adapter.eval()
        domain_adapter = domain_adapter.to(self.device)
        for param in domain_adapter.parameters():
            param.requires_grad = False
        
        # Create adapted YOLO model
        self.adapted_model = DomainAdaptedYOLO(
            yolo_model=yolo_model,
            feature_reducer=domain_adapter.feature_reducer
        ).to(self.device)
        
        # Ensure evaluation mode
        self.adapted_model.eval()
        for param in self.adapted_model.parameters():
            param.requires_grad = False

    def evaluate_frame_both_models(self, img_path, ground_truth, save_visualization=True):
        """
        Run both baseline and adapted models on the same frame and return predictions.
        """
        frame_number = int(os.path.basename(img_path).split('.')[0])
        video_name = os.path.basename(os.path.dirname(img_path))
        
        # Get baseline predictions using parent class method
        baseline_predictions = self.evaluate_frame(img_path, ground_truth, save_visualization)
        
        # Get adapted model predictions
        adapted_predictions = []
        
        try:
            # Load and preprocess image
            img = Image.open(img_path)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            # Run adapted model
            with torch.no_grad():
                predictions = self.adapted_model(img_tensor)  # This returns YOLO predictions
                
                # Process each detection
                for box in predictions.boxes:
                    instrument_class = int(box.cls)
                    confidence = float(box.conf)
                    
                    if confidence >= CONFIDENCE_THRESHOLD:
                        if instrument_class in IGNORED_INSTRUMENTS:
                            continue
                            
                        cholect50_instrument = TOOL_MAPPING.get(instrument_class)
                        if not cholect50_instrument:
                            continue
                        
                        mapped_instrument = CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING.get(cholect50_instrument)
                        if mapped_instrument:
                            prediction = {
                                'frame_id': f"{video_name}_frame{frame_number}",
                                'instrument': {
                                    'name': mapped_instrument,
                                    'confidence': confidence,
                                    'binary_pred': 1 if confidence >= CONFIDENCE_THRESHOLD else 0
                                }
                            }
                            adapted_predictions.append(prediction)
                    
        except Exception as e:
            print(f"Error processing adapted model for frame {frame_number}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return {
            'baseline': baseline_predictions,
            'adapted': adapted_predictions
        }

def analyze_label_distribution(dataset_dir, videos):
    """Analyzes the distribution of ground truth instrument labels"""
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
    """Prints a formatted report of the metrics"""
    print("\n====== INSTRUMENT DETECTION EVALUATION REPORT ======")
    print("=" * 70)
    print(f"{'Instrument':15s} {'F1-Score':>10s} {'Precision':>10s} {'Recall':>10s} "
          f"{'AP':>10s} {'Support':>10s} {'Predictions':>12s}")
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

class DetailedDebugYOLO(nn.Module):
    def __init__(self, yolo_path):
        super().__init__()
        
        # Load model like in original script
        self.original_model = YOLO(yolo_path)
        self.original_model.model.eval()
        self.model = self.original_model.model
        
        # Register hooks for layers 9 and 10
        self.layer_outputs = {}
        self.register_debug_hooks()
        
    def register_debug_hooks(self):
        def make_hook(layer_idx):
            def hook(module, input_feat, output_feat):
                # Detailed feature analysis
                self.layer_outputs[layer_idx] = {
                    'input': {
                        'type': type(input_feat),
                        'shapes': [f.shape if torch.is_tensor(f) else type(f) for f in input_feat],
                        'stats': [{
                            'min': f.min().item(),
                            'max': f.max().item(),
                            'mean': f.mean().item(),
                            'std': f.std().item()
                        } if torch.is_tensor(f) else None for f in input_feat]
                    },
                    'output': {
                        'type': type(output_feat),
                        'shape': output_feat.shape if torch.is_tensor(output_feat) else 
                                [o.shape for o in output_feat] if isinstance(output_feat, (tuple, list)) else type(output_feat),
                        'stats': {
                            'min': output_feat.min().item(),
                            'max': output_feat.max().item(),
                            'mean': output_feat.mean().item(),
                            'std': output_feat.std().item()
                        } if torch.is_tensor(output_feat) else None
                    }
                }
                return output_feat
            return hook

        # Register hooks for specific layers
        for i, layer in enumerate(self.model.model):
            if i in [8, 9, 10, 11]:  # Monitor layers around our target
                layer.register_forward_hook(make_hook(i))
                
    def compare_inference_modes(self, img_path):
        """Compare original YOLO inference with our modified version"""
        print("\n=== Comparing Inference Modes ===")
        
        # Original YOLO inference
        print("\n1. Original YOLO inference:")
        img = Image.open(img_path)
        with torch.no_grad():
            results = self.original_model(img)
            print(f"Type of results: {type(results)}")
            print(f"Results structure: {results}")
            
            # Analyze detections
            if hasattr(results[0], 'boxes'):
                boxes = results[0].boxes
                print(f"\nNumber of detections: {len(boxes)}")
                for i, box in enumerate(boxes):
                    print(f"\nDetection {i}:")
                    print(f"Box coordinates: {box.xyxy}")
                    print(f"Confidence: {box.conf}")
                    print(f"Class: {box.cls}")
        
        # Our modified inference
        print("\n2. Modified inference flow:")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            # Forward pass through model
            output = self.model(img_tensor)
            
            print("\nLayer-by-layer analysis:")
            for layer_idx, data in sorted(self.layer_outputs.items()):
                print(f"\nLayer {layer_idx}:")
                print("Input features:")
                print(f"- Types: {data['input']['type']}")
                print(f"- Shapes: {data['input']['shapes']}")
                if data['input']['stats'][0]:
                    stats = data['input']['stats'][0]
                    print(f"- Stats: min={stats['min']:.3f}, max={stats['max']:.3f}, "
                          f"mean={stats['mean']:.3f}, std={stats['std']:.3f}")
                
                print("\nOutput features:")
                print(f"- Type: {data['output']['type']}")
                print(f"- Shape: {data['output']['shape']}")
                if data['output']['stats']:
                    stats = data['output']['stats']
                    print(f"- Stats: min={stats['min']:.3f}, max={stats['max']:.3f}, "
                          f"mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
        return results

def debug_original_yolo(model_path, img_path):
    """Debug helper to understand original YOLO inference"""
    debugger = DetailedDebugYOLO(model_path)
    return debugger.compare_inference_modes(img_path)

def main():
    """Test domain-adapted YOLO model with feature reducer"""
    try:
        print("\n=== Initializing Models ===")
        # Initialize ModelLoader
        loader = ModelLoader()
        
        # Run debug analysis first
        print("\n=== Running Debug Analysis ===")
        test_frame_path = os.path.join(loader.dataset_path, "Videos", "VID08", "030300.png")
        debug_results = debug_original_yolo(str(loader.yolo_weights), test_frame_path)
        print("\n=== Debug Analysis Complete ===")
        
        # Load base YOLO model
        print("\nLoading base YOLO model...")
        yolo_model = load_yolo_model(str(loader.yolo_weights))
        yolo_model.model.eval()
        
        # Path to feature reducer weights
        weights_path = "/home/Bartscht/YOLO/surgical-instrument-action-detection/domain_adaptation/hei_chole/experiments/spatial_model_epoch_0.pt"
        
        print("\n=== Creating Domain-Adapted YOLO ===")
        # Initialize domain adapter
        domain_adapter = DomainAdapter(str(loader.yolo_weights))
        
        # Load feature reducer weights
        print(f"\nLoading feature reducer weights from: {weights_path}")
        checkpoint = torch.load(weights_path)
        domain_adapter.feature_reducer.load_state_dict(checkpoint['feature_reducer'])
        
        # Create adapted model
        adapted_yolo = DomainAdaptedYOLO(
            yolo_path=str(loader.yolo_weights),
            feature_reducer=domain_adapter.feature_reducer
        )
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        adapted_yolo = adapted_yolo.to(device)
        print(f"\nModel moved to device: {device}")
        
        # Test frames
        test_frames = [
            {'frame': '030300', 'path': "VID08"},
            {'frame': '030325', 'path': "VID08"},
            {'frame': '030350', 'path': "VID08"},
            {'frame': '030375', 'path': "VID08"}
        ]
        
        print("\n=== Testing Model on Frames ===")
        for frame_info in test_frames:
            frame_number = frame_info['frame']
            video_path = frame_info['path']
            
            # Construct frame path
            frame_path = os.path.join(
                loader.dataset_path, 
                "Videos", 
                video_path,
                f"{frame_number}.png"
            )
            
            print(f"\nProcessing frame {frame_number} from {video_path}")
            
            try:
                # Load and preprocess image
                img = Image.open(frame_path)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # Run inference
                with torch.no_grad():
                    print(f"\nRunning inference on frame {frame_number}")
                    predictions = adapted_yolo(img_tensor)
                    
                    # Print detailed prediction information
                    if isinstance(predictions, (tuple, list)):
                        for i, pred in enumerate(predictions):
                            print(f"Prediction {i} type: {type(pred)}")
                            print(f"Prediction {i} shape: {pred.shape if hasattr(pred, 'shape') else 'N/A'}")
                    else:
                        print(f"Prediction type: {type(predictions)}")
                        print(f"Prediction shape: {predictions.shape if hasattr(predictions, 'shape') else 'N/A'}")
                    
            except Exception as e:
                print(f"Error processing frame {frame_number}: {str(e)}")
                traceback.print_exc()
                continue
        
        print("\n=== Testing Complete ===")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        traceback.print_exc()
               
if __name__ == '__main__':
    main()