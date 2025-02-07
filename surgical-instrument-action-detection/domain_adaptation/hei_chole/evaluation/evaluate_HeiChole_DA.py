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
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results, Boxes
from ultralytics.utils.tal import make_anchors
from datetime import datetime

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
        
        # Load YOLO model
        model = YOLO(yolo_path)
        model.model.eval()
        model.predict(source=None, stream=True)
        self.yolo_model = model.model
        self.detector = self.yolo_model.model[-1]  # Original Detection Head
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to correct device
        self.yolo_model = self.yolo_model.to(self.device)
        
        self.feature_reducer = feature_reducer.to(self.device)
        self.feature_layer = 9  # SPPF layer
        
        # Dictionary for Skip-Connections
        self.saved_features = {}
        
        # Layer mapping from YAML
        self.skip_connections = {
            12: [11, 6],    # [Upsampled features, Backbone P4]
            15: [14, 4],    # [Upsampled features, Backbone P3]
            18: [17, 13],   # [Downsampled features, Head P4]
            21: [20, 10]    # [Downsampled features, Head P5]
        }
        
        # Register hooks
        for i, layer in enumerate(self.yolo_model.model):
            if i == self.feature_layer:
                layer.register_forward_hook(self._feature_modification_hook)
            layer.register_forward_hook(self._make_save_hook(i))
        
        # Ensure eval mode
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
            
    def _feature_modification_hook(self, module, input_feat, output_feat):
        """Hook for Feature Modification after SPPF"""
        try:
            main_features = input_feat[0]
            modified = self.feature_reducer(main_features)
            
            # Debug info
            print(f"\nFeature Modification at SPPF:")
            print(f"Input shape: {main_features.shape}")
            print(f"Modified shape: {modified.shape}")
            print(f"Original stats - min: {output_feat.min():.3f}, max: {output_feat.max():.3f}, "
                  f"mean: {output_feat.mean():.3f}, std: {output_feat.std():.3f}")
            print(f"Modified stats - min: {modified.min():.3f}, max: {modified.max():.3f}, "
                  f"mean: {modified.mean():.3f}, std: {modified.std():.3f}")
            
            return modified
            
        except Exception as e:
            print(f"Error in feature modification: {str(e)}")
            traceback.print_exc()
            return output_feat
            
    def _make_save_hook(self, layer_idx):
        """Creates a Hook to save Features for Skip-Connections"""
        def hook(module, input_feat, output_feat):
            try:
                # Save features for later Skip-Connections
                self.saved_features[layer_idx] = output_feat
                
                # Debug info
                print(f"\nLayer {layer_idx} ({type(module).__name__}):")
                print(f"Output shape: {output_feat.shape if torch.is_tensor(output_feat) else type(output_feat)}")
                
                # Handle Concat Layers
                if layer_idx in self.skip_connections:
                    # Get features to concatenate
                    features_to_concat = []
                    for idx in self.skip_connections[layer_idx]:
                        if idx in self.saved_features:
                            feat = self.saved_features[idx]
                            if isinstance(feat, torch.Tensor):
                                features_to_concat.append(feat)
                    
                    if len(features_to_concat) > 0:
                        # Concatenate along channel dimension
                        output_feat = torch.cat(features_to_concat, dim=1)
                        print(f"Concat at layer {layer_idx} - New shape: {output_feat.shape}")
                
                return output_feat
                
            except Exception as e:
                print(f"Error in save hook at layer {layer_idx}: {str(e)}")
                traceback.print_exc()
                return output_feat
                
        return hook
    
    def forward(self, x):
        """Forward pass with Skip-Connections"""
        with torch.no_grad():
            try:
                # Clear saved features
                self.saved_features.clear()
                
                # Ensure input tensor format
                if not isinstance(x, torch.Tensor):
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )
                    ])
                    x = transform(x).unsqueeze(0)
                
                x = x.to(self.device)
                
                # Run model and get the output tensor [1, 11, 5376]
                raw_output = self.yolo_model(x)
                
                print("\nRaw output type:", type(raw_output))
                print("Raw output length:", len(raw_output) if hasattr(raw_output, '__len__') else "N/A")
                
                # Identify the first tensor
                if isinstance(raw_output, tuple):
                    first_tensor = raw_output[0]
                    print("First tensor shape:", first_tensor.shape)
                else:
                    first_tensor = raw_output
                
                # Create Results object to match original inference
                results = Results(
                    orig_img=x.cpu().numpy().squeeze(),
                    path='',  # You can set an actual path if needed
                    names=TOOL_MAPPING  # Use original model's class names
                )
                
                # Use detector's postprocess to get boxes
                processed_output = self.detector.postprocess(
                    first_tensor.permute(0, 2, 1),
                    max_det=300,
                    nc=self.detector.nc
                )
                
                # Create Boxes object
                results.boxes = Boxes(processed_output[0], x.shape[-2:])
                print("\nProcessed Results Details (Confidence > 0.14):")
                high_conf_detections = []
                for i, (box, conf, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
                    if conf.item() > 0.14:
                        print(f"\nDetection {i}:")
                        print(f"Box coordinates: {box}")
                        print(f"Confidence: {conf.item()}")
                        print(f"Class: {cls.item()}")
                        print(f"Class name: {TOOL_MAPPING[int(cls)]}")
                        high_conf_detections.append((box, conf, cls))
            
                # If no high confidence detections, print a message
                if not high_conf_detections:
                    print("No detections with confidence > 0.15")
                
                return [results]
                
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

    def evaluate_frame(self, img_path, ground_truth, save_visualization=False):
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
        
        # Move model to correct device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
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
                        'device': [f.device if torch.is_tensor(f) else None for f in input_feat],
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
                        'device': output_feat.device if torch.is_tensor(output_feat) else None,
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
                
    def analyze_model_structure(self):
        """Analyze the model structure in detail"""
        print("\nModel Structure Analysis:")
        for i, layer in enumerate(self.model.model):
            if i in [8, 9, 10, 11]:
                print(f"\nLayer {i}: {type(layer).__name__}")
                print(f"Parameters: {sum(p.numel() for p in layer.parameters())}")
                for name, param in layer.named_parameters():
                    print(f"- {name}: shape={param.shape}, device={param.device}")
                
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
        
        # Modified inference flow
        print("\n2. Modified inference flow:")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)  # Move tensor to correct device
        
        with torch.no_grad():
            # First, analyze model structure
            self.analyze_model_structure()
            
            # Forward pass through model
            output = self.model(img_tensor)
            
            print("\nLayer-by-layer analysis:")
            for layer_idx, data in sorted(self.layer_outputs.items()):
                print(f"\nLayer {layer_idx}:")
                print("Input features:")
                print(f"- Types: {data['input']['type']}")
                print(f"- Shapes: {data['input']['shapes']}")
                print(f"- Devices: {data['input']['device']}")
                if data['input']['stats'][0]:
                    stats = data['input']['stats'][0]
                    print(f"- Stats: min={stats['min']:.3f}, max={stats['max']:.3f}, "
                          f"mean={stats['mean']:.3f}, std={stats['std']:.3f}")
                
                print("\nOutput features:")
                print(f"- Type: {data['output']['type']}")
                print(f"- Shape: {data['output']['shape']}")
                print(f"- Device: {data['output']['device']}")
                if data['output']['stats']:
                    stats = data['output']['stats']
                    print(f"- Stats: min={stats['min']:.3f}, max={stats['max']:.3f}, "
                          f"mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
        return results
    
def debug_original_yolo(model_path, img_path):
    """Debug helper to understand original YOLO inference"""
    debugger = DetailedDebugYOLO(model_path)
    return debugger.compare_inference_modes(img_path)

def test_domain_adapted_yolo(yolo_path, feature_reducer, test_image):
    """Test function for the domain adapted YOLO"""
    try:
        # Initialize model
        model = DomainAdaptedYOLO(yolo_path, feature_reducer)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Load and preprocess test image
        img = Image.open(test_image)
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
            results = model(img_tensor)
            
            # Process results
            if hasattr(results[0], 'boxes'):
                boxes = results[0].boxes
                print(f"Number of detections: {len(boxes)}")
                for i, box in enumerate(boxes):
                    print(f"\nDetection {i}:")
                    print(f"Box coordinates: {box.xyxy}")
                    print(f"Confidence: {box.conf}")
                    print(f"Class: {box.cls}")
            
        return results
        
    except Exception as e:
        print(f"Error in test: {str(e)}")
        traceback.print_exc()
        return None

class xYOLODebugger:
    def __init__(self, yolo_path):
        self.model = YOLO(yolo_path)
        self.model.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def inspect_model_structure(self):
        """Inspiziert die Struktur des YOLO Models"""
        print("\nYOLO Model Structure:")
        print("=====================")
        print(f"Type of model: {type(self.model)}")
        print(f"Type of model.model: {type(self.model.model)}")
        print(f"Type of model.predictor: {type(self.model.predictor)}")
        
        # Untersuche die predict Methode
        print("\nPredict Method Source:")
        print("=====================")
        import inspect
        print(inspect.getsource(self.model.predict))
        
    def trace_prediction_flow(self, img_path):
        """Verfolgt den Vorhersagefluss von Input bis Results"""
        print("\nPrediction Flow Analysis:")
        print("=====================")
        
        # 1. Input Verarbeitung
        img = Image.open(img_path)
        print(f"\n1. Original Image Type: {type(img)}")
        
        # 2. Modell Forward Pass
        with torch.no_grad():
            # Standard YOLO Predict
            results1 = self.model(img)
            print(f"\n2a. Standard YOLO Results: {type(results1)}")
            print(f"Results Structure: {results1[0].__dict__.keys()}")
            
            # Manueller Forward Pass
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            outputs = self.model.model(img_tensor)
            print(f"\n2b. Raw Model Output Type: {type(outputs)}")
            print(f"Output Shapes: {[output.shape for output in outputs if isinstance(output, torch.Tensor)]}")
            
            # 3. Results Erzeugung
            print("\n3. Results Creation Process:")
            # Get predictor instance
            predictor = self.model.predictor
            if predictor is None:
                predictor = BasePredictor()
            
            # Untersuche postprocess
            print("\nPostprocess Method:")
            if hasattr(predictor, 'postprocess'):
                results2 = predictor.postprocess(outputs, img)
                print(f"Post-processed Results Type: {type(results2)}")
                if isinstance(results2, list) and len(results2) > 0:
                    print(f"First Result Keys: {results2[0].__dict__.keys()}")
            
    def analyze_results_creation(self, img_path):
        """Analysiert wie Results-Objekte erstellt werden"""
        img = Image.open(img_path)
        results = self.model(img)
        result = results[0]  # Nimm erstes Result
        
        print("\nResults Object Analysis:")
        print("=====================")
        print(f"Result Type: {type(result)}")
        print(f"Available Attributes: {result.__dict__.keys()}")
        
        if hasattr(result, 'boxes'):
            boxes = result.boxes
            print("\nBoxes Analysis:")
            print(f"Boxes Type: {type(boxes)}")
            print(f"Box Attributes: {boxes.__dict__.keys()}")
            if len(boxes) > 0:
                print("\nFirst Box Details:")
                print(f"Coordinates: {boxes.xyxy[0]}")
                print(f"Confidence: {boxes.conf[0]}")
                print(f"Class: {boxes.cls[0]}")
   
class YOLODebugger:
    def __init__(self, yolo_path):
        self.model = YOLO(yolo_path)  # Hier sollte idealerweise dein DomainAdaptedYOLO verwendet werden!
        self.model.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.model = self.model.model.to(self.device)
        # Bei DomainAdaptedYOLO sollten die saved_features bereits durch die Hooks befüllt werden.
        # Der Detection Head (Detect Layer) ist im letzten Layer des Modells enthalten:
        self.detector = self.model.model.model[-1]  
        print("\nDetector type:", type(self.detector))
        
    def reconstruct_feature_maps(self, output, strides=[8, 16, 32]):
        """
        Rekonstruiert Feature-Maps aus dem zusammengefassten Tensor.
        Diese Methode wird nicht mehr für die Domain Adaptation benötigt, da wir direkt
        die gespeicherten Features aus den Schichten 16, 19, 22 verwenden.
        """
        batch_size, num_anchors, features = output.shape
        total_pixels = features // len(strides)
        feature_maps = []
        start_idx = 0
        for stride in strides:
            grid_size = int(np.sqrt(total_pixels / (stride/8)))
            pixels = grid_size * grid_size
            end_idx = start_idx + pixels
            features_stride = output[:, :, start_idx:end_idx]
            feature_map = features_stride.view(batch_size, num_anchors, grid_size, grid_size)
            feature_maps.append(feature_map)
            start_idx = end_idx
        return feature_maps

    def trace_prediction_flow(self, img_path):
        """
        Verfolgt und analysiert den Vorhersagefluss von Input bis Results.
        Im modified flow nutzen wir die in DomainAdaptedYOLO gespeicherten Features (z.B. Layer 16, 19, 22)
        und injizieren sie in den vortrainierten Detection Head.
        """
        print("\nPrediction Flow Analysis:")
        print("=====================")
        
        img = Image.open(img_path)
        
        with torch.no_grad():
            # 1. Original YOLO inference (Standard-Pipeline)
            print("\n1. Original YOLO inference:")
            results1 = self.model(img)
            self._analyze_results("Original Results", results1)
            
            # 2. Modified inference flow: Wir nehmen an, dass unser DomainAdaptedYOLO-Objekt
            # über den saved_features-Dictionary verfügt, in dem die Features der relevanten Schichten
            # (z.B. 16, 19 und 22) gespeichert wurden.
            print("\n2. Modified inference flow:")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            # Lasse den Forward-Pass laufen, sodass die Hooks die saved_features füllen:
            _ = self.model.model(img_tensor)
            
            try:
                # Sammle die Feature-Maps, die der Detection Head benötigt:
                # Laut YAML werden Layer 16, 19 und 22 als Inputs für den Detect-Layer verwendet.
                required_layers = [16, 19, 22]
                features = []
                for idx in required_layers:
                    if idx in self.model.saved_features:
                        features.append(self.model.saved_features[idx])
                    else:
                        raise ValueError(f"Saved feature for layer {idx} nicht gefunden.")
                # Optional: Überprüfe die Shapes der Features
                print("\nCollected Features for Detect Layer:")
                for i, feat in enumerate(features):
                    print(f"Layer {required_layers[i]} shape: {feat.shape}")
                
                # Nun injizieren wir diese Features in den Detection Head.
                # Der Detection Head verfügt über die Methode _inference, die intern den Postprocessing-Flow ausführt.
                preds = self.detector._inference(features)
                
                print("\nDetector Inference Output:")
                print(f"Shape: {preds.shape}")
                print(f"Min: {preds.min().item():.3f}")
                print(f"Max: {preds.max().item():.3f}")
                print(f"Mean: {preds.mean().item():.3f}")
                
                # Erzeuge ein Results-Objekt, um die finalen Vorhersagen zu visualisieren
                result = Results(
                    orig_img=img_tensor.cpu().numpy(),
                    path=img_path,
                    names=self.model.model.names
                )
                result.boxes = Boxes(preds[0], img_tensor.shape[-2:])
                
                print("\nFinal Modified Results:")
                self._analyze_results("Modified Results", [result])
            except Exception as e:
                print(f"\nError during modified inference flow: {str(e)}")
                traceback.print_exc()

    def _analyze_results(self, title, results):
        """Analysiert YOLO Results"""
        if not isinstance(results, list) or not results:
            return
        result = results[0]
        print(f"\n=== {title} ===")
        print(f"Type: {type(results)}")
        print(f"Structure: {result}")
        
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            print(f"\nNumber of detections: {len(result.boxes)}")
            boxes = result.boxes
            for i, (box, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
                print(f"\nDetection {i}:")
                print(f"Box coordinates: {box}")
                print(f"Confidence: {conf.item()}")
                print(f"Class: {cls.item()}")
                if hasattr(result, 'names'):
                    print(f"Class name: {result.names[int(cls)]}")
        if hasattr(result, 'speed'):
            print("\nProcessing Speed:")
            for key, value in result.speed.items():
                print(f"{key}: {value}ms")

class ComparativeHeiCholeEvaluator:
    def __init__(self, baseline_model, domain_adapted_model, dataset_dir, 
                 baseline_threshold=0.5, adapted_threshold=0.1):
        """
        Initialize evaluator for comparing baseline and domain-adapted models.
        
        Args:
            baseline_model: Original YOLO model
            domain_adapted_model: Domain-adapted YOLO model
            dataset_dir: Path to dataset
            baseline_threshold: Confidence threshold for baseline model (default: 0.5)
            adapted_threshold: Confidence threshold for adapted model (default: 0.1)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.baseline_model = baseline_model
        self.domain_adapted_model = domain_adapted_model
        self.dataset_dir = dataset_dir
        self.baseline_threshold = baseline_threshold
        self.adapted_threshold = adapted_threshold
        
        # Separate metric calculators for each model
        self.baseline_metrics = BinaryMetricsCalculator(confidence_threshold=baseline_threshold)
        self.adapted_metrics = BinaryMetricsCalculator(confidence_threshold=adapted_threshold)

    def evaluate_frame_both_models(self, img_path, ground_truth, save_visualization=False):
        """
        Evaluate a single frame with both models and return comparative results.
        """
        frame_number = int(os.path.basename(img_path).split('.')[0])
        video_name = os.path.basename(os.path.dirname(img_path))
        
        img = Image.open(img_path)
        
        # Create two copies for visualization
        baseline_img = img.copy()
        adapted_img = img.copy()
        
        baseline_draw = ImageDraw.Draw(baseline_img)
        adapted_draw = ImageDraw.Draw(adapted_img)
        
        predictions = {
            'baseline': [],
            'adapted': []
        }
        
        try:
            # Process with baseline model
            baseline_results = self.baseline_model(img)
            predictions['baseline'] = self._process_detections(
                baseline_results[0].boxes,
                frame_number,
                video_name,
                baseline_draw if save_visualization else None,
                is_adapted=False
            )
            
            # Process with domain-adapted model
            adapted_results = self.domain_adapted_model(img)
            predictions['adapted'] = self._process_detections(
                adapted_results[0].boxes,
                frame_number,
                video_name,
                adapted_draw if save_visualization else None,
                is_adapted=True
            )
            
            # Save visualizations
            if save_visualization:
                viz_dir = os.path.join(self.dataset_dir, "visualizations", "comparison")
                os.makedirs(viz_dir, exist_ok=True)
                
                baseline_img.save(os.path.join(viz_dir, f"{video_name}_frame{frame_number}_baseline.png"))
                adapted_img.save(os.path.join(viz_dir, f"{video_name}_frame{frame_number}_adapted.png"))
                
                # Create side-by-side comparison
                comparison = Image.new('RGB', (baseline_img.width * 2, baseline_img.height))
                comparison.paste(baseline_img, (0, 0))
                comparison.paste(adapted_img, (baseline_img.width, 0))
                comparison.save(os.path.join(viz_dir, f"{video_name}_frame{frame_number}_comparison.png"))
            
            return predictions
            
        except Exception as e:
            print(f"Error processing frame {frame_number}: {str(e)}")
            return predictions

    def _process_detections(self, boxes, frame_number, video_name, draw=None, is_adapted=False):
        """
        Process detections from either model.
        
        Args:
            boxes: Detection boxes from model
            frame_number: Current frame number
            video_name: Name of video
            draw: ImageDraw object for visualization
            is_adapted: True if processing adapted model output, False for baseline
        """
        predictions = []
        
        for detection in boxes:
            instrument_class = int(detection.cls)
            confidence = float(detection.conf)
            
            threshold = self.adapted_threshold if is_adapted else self.baseline_threshold
            if confidence >= threshold:
                if instrument_class in IGNORED_INSTRUMENTS:
                    continue
                
                try:
                    cholect50_instrument = TOOL_MAPPING[instrument_class]
                except KeyError:
                    continue
                
                mapped_instrument = CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING.get(cholect50_instrument)
                
                if mapped_instrument:
                    if draw:
                        box = detection.xyxy[0]
                        x1, y1, x2, y2 = map(int, box)
                        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                        text = f"{mapped_instrument}\nConf: {confidence:.2f}"
                        draw.text((x1, y1-20), text, fill='blue')
                    
                    predictions.append({
                        'frame_id': f"{video_name}_frame{frame_number}",
                        'instrument': {
                            'name': mapped_instrument,
                            'confidence': confidence,
                            'binary_pred': 1 if confidence >= CONFIDENCE_THRESHOLD else 0
                        }
                    })
        
        return predictions

    def evaluate_dataset(self, videos_to_analyze):
        """
        Evaluate entire dataset with both models.
        """
        print("\nStarting comparative evaluation...")
        
        results = {
            'baseline': {'predictions': {}, 'metrics': None},
            'adapted': {'predictions': {}, 'metrics': None}
        }
        
        total_frames = 0
        ground_truth = {}
        
        for video in videos_to_analyze:
            print(f"\nProcessing video: {video}")
            
            # Load ground truth
            gt = self.load_ground_truth(video)
            ground_truth.update(gt)
            
            # Process frames
            video_folder = os.path.join(self.dataset_dir, "Videos", video)
            for frame_file in tqdm(os.listdir(video_folder)):
                if frame_file.endswith('.png'):
                    total_frames += 1
                    frame_id = f"{video}_frame{frame_file.split('.')[0]}"
                    img_path = os.path.join(video_folder, frame_file)
                    
                    # Get predictions from both models
                    frame_predictions = self.evaluate_frame_both_models(
                        img_path,
                        gt[int(frame_file.split('.')[0])],
                        save_visualization=(total_frames % 100 == 0)  # Save every 100th frame
                    )
                    
                    # Store predictions
                    if frame_predictions['baseline']:
                        results['baseline']['predictions'][frame_id] = frame_predictions['baseline']
                    if frame_predictions['adapted']:
                        results['adapted']['predictions'][frame_id] = frame_predictions['adapted']
        
        # Calculate metrics with different thresholds
        results['baseline']['metrics'] = self.baseline_metrics.calculate_metrics(
            results['baseline']['predictions'],
            ground_truth
        )
        results['adapted']['metrics'] = self.adapted_metrics.calculate_metrics(
            results['adapted']['predictions'],
            ground_truth
        )
        
        # Log thresholds used
        print(f"\nConfidence Thresholds Used:")
        print(f"Baseline Model: {self.baseline_threshold}")
        print(f"Adapted Model:  {self.adapted_threshold}")
        
        return results, total_frames, ground_truth

    def print_comparative_report(self, results, total_frames):
        """Print detailed comparative report of both models."""
        print("\n====== COMPARATIVE EVALUATION REPORT ======")
        print("=" * 70)
        
        for model_type in ['baseline', 'adapted']:
            print(f"\n{model_type.upper()} MODEL RESULTS:")
            print("-" * 50)
            metrics = results[model_type]['metrics']
            
            print(f"{'Instrument':15s} {'F1-Score':>10s} {'Precision':>10s} {'Recall':>10s} "
                  f"{'AP':>10s} {'Support':>10s} {'Predictions':>12s}")
            print("-" * 70)
            
            for label, scores in metrics['per_class'].items():
                pred_count = scores['predictions']
                percentage = (pred_count / total_frames) * 100 if total_frames > 0 else 0
                
                print(f"{label:15s} {scores['f1_score']:10.4f} {scores['precision']:10.4f} "
                      f"{scores['recall']:10.4f} {scores['ap_score']:10.4f} {scores['support']:10d} "
                      f"{pred_count:8d} ({percentage:5.1f}%)")
            
            print("\nMean Metrics:")
            means = metrics['mean_metrics']
            print(f"Mean AP:        {means['mean_ap']:.4f}")
            print(f"Mean F1-Score:  {means['mean_f1']:.4f}")
            print(f"Mean Precision: {means['mean_precision']:.4f}")
            print(f"Mean Recall:    {means['mean_recall']:.4f}")
        
        # Print improvement analysis
        self._print_improvement_analysis(results)

    def _print_improvement_analysis(self, results):
        """Analyze and print improvements of adapted model over baseline."""
        print("\n====== IMPROVEMENT ANALYSIS ======")
        print("=" * 40)
        
        baseline_metrics = results['baseline']['metrics']
        adapted_metrics = results['adapted']['metrics']
        
        # Compare per-class metrics
        print("\nPer-Class Improvements:")
        print("-" * 30)
        
        for label in baseline_metrics['per_class'].keys():
            baseline = baseline_metrics['per_class'][label]
            adapted = adapted_metrics['per_class'][label]
            
            f1_diff = adapted['f1_score'] - baseline['f1_score']
            ap_diff = adapted['ap_score'] - baseline['ap_score']
            
            print(f"\n{label}:")
            print(f"F1-Score: {f1_diff:+.4f}")
            print(f"AP Score: {f1_diff:+.4f}")
        
        # Compare mean metrics
        print("\nMean Metric Improvements:")
        print("-" * 30)
        
        mean_metrics = [
            ('mean_ap', 'Mean AP'),
            ('mean_f1', 'Mean F1'),
            ('mean_precision', 'Mean Precision'),
            ('mean_recall', 'Mean Recall')
        ]
        
        for metric_key, metric_name in mean_metrics:
            diff = adapted_metrics['mean_metrics'][metric_key] - baseline_metrics['mean_metrics'][metric_key]
            print(f"{metric_name}: {diff:+.4f}")

    def load_ground_truth(self, video):
        """Load ground truth annotations for a video."""
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
                        frame_annotations[frame_number]['instruments'][instr_name] = 1 if present > 0 else 0
                
                return frame_annotations
                
        except Exception as e:
            print(f"Error loading annotations: {str(e)}")
            raise

def main():
    """Compare baseline and domain-adapted models on HeiChole dataset"""
    try:
        print("\n=== Initializing Models ===")
        # Initialize ModelLoader
        loader = ModelLoader()
        
        # Load baseline YOLO model correctly
        print("\nLoading baseline YOLO model...")
        yolo_model = load_yolo_model(str(loader.yolo_weights))  # Using your existing load_yolo_model function
        
        # Create and load domain adapter
        print("\nInitializing domain adapter...")
        domain_adapter = DomainAdapter(str(loader.yolo_weights))
        
        # Load feature reducer weights
        feature_reducer_path = Path("/home/Bartscht/YOLO/surgical-instrument-action-detection/domain_adaptation/hei_chole/experiments/class_aware_adapter_weights_class_aware_domain_adaptation/class_aware_feature_reducer.pt")
        print(f"\nLoading feature reducer from: {feature_reducer_path}")
        
        if not feature_reducer_path.exists():
            raise FileNotFoundError(f"Feature reducer weights not found at: {feature_reducer_path}")
            
        checkpoint = torch.load(feature_reducer_path)
        domain_adapter.feature_reducer.load_state_dict(checkpoint['state_dict'])
        
        # Create domain-adapted YOLO model
        print("\nCreating domain-adapted YOLO model...")
        adapted_model = DomainAdaptedYOLO(
            yolo_path=str(loader.yolo_weights),
            feature_reducer=domain_adapter.feature_reducer
        )
        
        dataset_dir = str(loader.dataset_path)
        
        # Specify videos to analyze
        videos_to_analyze = ["VID08", "VID13"]
        print(f"\nAnalyzing videos: {', '.join(videos_to_analyze)}")
        
        # Create comparative evaluator with different thresholds
        print("\nInitializing comparative evaluator...")
        evaluator = ComparativeHeiCholeEvaluator(
            baseline_model=yolo_model,
            domain_adapted_model=adapted_model,
            dataset_dir=dataset_dir,
            baseline_threshold=0.5,    # Higher threshold for baseline
            adapted_threshold=0.1      # Lower threshold for adapted model
        )
        
        # First analyze ground truth distribution
        print("\n=== Analyzing Ground Truth Distribution ===")
        gt_distribution = analyze_label_distribution(dataset_dir, videos_to_analyze)
        
        # Run evaluation
        print("\n=== Starting Model Evaluation ===")
        results, total_frames, ground_truth = evaluator.evaluate_dataset(videos_to_analyze)
        
        # Save results
        print("\n=== Saving Results ===")
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save metrics to JSON
        metrics_file = results_dir / "comparative_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'baseline': results['baseline']['metrics'],
                'adapted': results['adapted']['metrics'],
                'total_frames': total_frames,
                'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=4)
            
        print(f"\nMetrics saved to: {metrics_file}")
        
        # Print comparative report
        print("\n=== Generating Comparative Report ===")
        evaluator.print_comparative_report(results, total_frames)
        
        print("\n=== Evaluation Complete ===")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

def oldmain():
    """Test YOLO model debugging"""
    try:
        print("\n=== Initializing Models ===")
        loader = ModelLoader()
        
        debugger = YOLODebugger(str(loader.yolo_weights))
        
        # Test frame analysis
        test_frame_path = os.path.join(loader.dataset_path, "Videos", "VID08", "030300.png")
        
        # Add this block to print DomainAdaptedYOLO output
        print("\n=== Testing DomainAdaptedYOLO Output ===")
        # Load domain adapter
        domain_adapter = DomainAdapter(str(loader.yolo_weights))
        
        # Create domain-adapted YOLO
        adapted_yolo = DomainAdaptedYOLO(
            yolo_path=str(loader.yolo_weights),
            feature_reducer=domain_adapter.feature_reducer
        )
        
        # Prepare image
        img = Image.open(test_frame_path)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        img_tensor = transform(img).unsqueeze(0)
        
        # Run inference and print detailed output
        with torch.no_grad():
            output = adapted_yolo(img_tensor)
            print("\nDomainAdaptedYOLO Output:")
            print(f"Output Type: {type(output)}")
            
            # If output is a tensor or has 'boxes' attribute
            if hasattr(output, 'boxes'):
                boxes = output.boxes
                print(f"Number of detections: {len(boxes)}")
                for i, box in enumerate(boxes):
                    print(f"\nDetection {i}:")
                    print(f"Box coordinates: {box.xyxy}")
                    print(f"Confidence: {box.conf}")
                    print(f"Class: {box.cls}")
            elif isinstance(output, torch.Tensor):
                print(f"Tensor Shape: {output.shape}")
                print(f"Tensor Stats - Min: {output.min()}, Max: {output.max()}, Mean: {output.mean()}")
        
        # Optional: continue with original debugger trace
        debugger.trace_prediction_flow(test_frame_path)
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        traceback.print_exc()

def xmain():
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
        weights_path = "/home/Bartscht/YOLO/surgical-instrument-action-detection/domain_adaptation/hei_chole/experiments/class_aware_adapter_weights_class_aware_domain_adaptation/class_aware_feature_reducer.pt"
        
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
        #comparer = YOLOOutputComparer(str(loader.yolo_weights))
        #comparer.set_domain_adapter(domain_adapter, weights_path)

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

def yolomain():
    """Test YOLO model debugging"""
    try:
        print("\n=== Initializing Models ===")
        loader = ModelLoader()
        
        # Initialize debugger
        debugger = YOLODebugger(str(loader.yolo_weights))
        
        # Test frame analysis
        test_frame_path = os.path.join(loader.dataset_path, "Videos", "VID08", "030300.png")
        debugger.trace_prediction_flow(test_frame_path)
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    main()