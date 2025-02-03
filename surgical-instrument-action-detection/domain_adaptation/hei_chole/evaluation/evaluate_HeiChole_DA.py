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
        # YOLO Initialization
        model = YOLO(yolo_path)
        model.model.eval()
        model.predict(source=None, stream=True)  # Force inference mode
        self.yolo_model = model.model.model
        self.feature_layer = 8
        
        # Disable YOLO training
        for param in self.yolo_model.parameters():
            param.requires_grad = False
            
        # Feature Reducer that maintains spatial dimensions
        self.feature_reducer = nn.Sequential(
            # 1x1 Convolution zur subtilen Feature-Transformation
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Residual Block für Informationserhalt
            ResidualBlock(512, 512),
            
            # Gruppierte Convolution für effiziente Feature-Extraktion
            nn.Conv2d(512, 512, 
                      kernel_size=3, 
                      padding=1, 
                      groups=32),  # Gruppierte Convolution
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Ensure evaluation mode
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

       
class xDomainAdaptedYOLO(nn.Module):
    def __init__(self, yolo_model, feature_reducer):
        super().__init__()
        self.yolo_model = yolo_model.model
        self.feature_layer = 8
        
        # Store feature reducer
        self.feature_reducer = feature_reducer
        
        # Split YOLO layers
        self.pre_feature_layers = nn.ModuleList()
        self.post_feature_layers = nn.ModuleList()
        
        # Split the model at feature_layer
        for i, layer in enumerate(self.yolo_model.model):
            if i <= self.feature_layer:
                self.pre_feature_layers.append(layer)
            else:
                self.post_feature_layers.append(layer)
        
        # Ensure evaluation mode
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        with torch.no_grad():
            # Store intermediates for skip connections
            features = []
            
            # Pre-feature extraction (up to layer 8)
            for layer in self.pre_feature_layers:
                if isinstance(layer, (nn.modules.upsampling.Upsample, 
                                   ultralytics.nn.modules.conv.Concat)):
                    if isinstance(layer, ultralytics.nn.modules.conv.Concat):
                        x = torch.cat([x] + features[-layer.d:], 1)
                    else:
                        x = layer(x)
                else:
                    x = layer(x)
                features.append(x)
            
            # Apply domain adaptation at layer 8
            x = self.feature_reducer(x)
            features[-1] = x  # Update the last feature map
            
            # Post-feature processing (after layer 8)
            for layer in self.post_feature_layers:
                if isinstance(layer, ultralytics.nn.modules.conv.Concat):
                    x = torch.cat([x] + features[-layer.d:], 1)
                elif isinstance(layer, nn.modules.upsampling.Upsample):
                    x = layer(x)
                else:
                    x = layer(x)
                features.append(x)
            
            return x

class DebugC3k2(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.c3k2 = layer
        
    def forward(self, x):
        print(f"\nC3k2 Debug Info:")
        print(f"Input shape: {x.shape}")
        
        # Inspiziere die Conv Layer Gewichte
        if hasattr(self.c3k2, 'cv1'):
            w = self.c3k2.cv1.conv.weight
            print(f"First conv weight shape: {w.shape}")
            
        if hasattr(self.c3k2, 'cv2'):
            w = self.c3k2.cv2.conv.weight
            print(f"Second conv weight shape: {w.shape}")
            
        # Track internal transformations
        if hasattr(self.c3k2, 'cv1'):
            conv1_out = self.c3k2.cv1(x)
            print(f"After first conv: {conv1_out.shape}")
            
            # Wenn es ein chunk gibt, zeige die Dimensionen
            chunks = conv1_out.chunk(2, 1)
            print(f"After chunking: {[c.shape for c in chunks]}")
        
        return self.c3k2(x)

class DomainAdaptedYOLO(nn.Module):
    def __init__(self, yolo_model, feature_reducer):
        super().__init__()
        self.yolo_model = yolo_model.model
        self.feature_layer = 8
        
        # Store feature reducer
        self.feature_reducer = feature_reducer
        
        # Split YOLO layers
        self.pre_feature_layers = nn.ModuleList()
        self.post_feature_layers = nn.ModuleList()
        
        # Channel adapters for maintaining compatibility
        self.channel_adapters = nn.ModuleDict({
            # Key adapters for critical layers
            'adapt_19': nn.Sequential(
                nn.Conv2d(512, 768, kernel_size=1),
                nn.BatchNorm2d(768),
                nn.ReLU()
            ),
            'adapt_20': nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=1),
                nn.BatchNorm2d(1024),
                nn.ReLU()
            )
        })
        
        # Analyze and split the model
        self._analyze_and_split_model()
        
        # Ensure evaluation mode
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
            
    def _analyze_and_split_model(self):
        """Analyzes the YOLO model architecture and splits it at the feature layer"""
        print("\nAnalyzing YOLO architecture...")
        
        features = []
        x = torch.randn(1, 3, 512, 512)  # Dummy input for analysis
        
        for i, layer in enumerate(self.yolo_model.model):
            # Split layers at feature_layer
            if i <= self.feature_layer:
                self.pre_feature_layers.append(layer)
            else:
                self.post_feature_layers.append(layer)
                
            # Analyze layer for debugging
            if hasattr(layer, 'conv'):
                print(f"\nLayer {i} Analysis:")
                print(f"Type: {type(layer).__name__}")
                print(f"Input channels: {layer.conv.in_channels}")
                print(f"Output channels: {layer.conv.out_channels}")
                
    def _handle_concat_layer(self, x, layer, features, layer_idx):
        """Handles concatenation layers with proper dimensionality"""
        if isinstance(layer, ultralytics.nn.modules.conv.Concat):
            concat_inputs = [x]
            for idx in range(layer.d):
                feature = features[-(idx + 1)]
                concat_inputs.append(feature)
            
            # Debug concatenation
            print(f"\nConcatenation at layer {layer_idx}:")
            print(f"Input shapes: {[f.shape for f in concat_inputs]}")
            
            return torch.cat(concat_inputs, 1)
        return x
    
    def _apply_channel_adaptation(self, x, layer_idx):
        """Applies channel adaptation where needed"""
        adapter_key = f'adapt_{layer_idx}'
        if adapter_key in self.channel_adapters:
            original_shape = x.shape
            x = self.channel_adapters[adapter_key](x)
            print(f"\nApplied channel adaptation at layer {layer_idx}")
            print(f"Shape changed from {original_shape} to {x.shape}")
        return x
    
    def forward(self, x):
        with torch.no_grad():
            # Store intermediate features for skip connections
            features = []
            
            # Pre-feature extraction (up to layer 8)
            for i, layer in enumerate(self.pre_feature_layers):
                if isinstance(layer, ultralytics.nn.modules.conv.Concat):
                    x = self._handle_concat_layer(x, layer, features, i)
                else:
                    x = layer(x)
                features.append(x)
                
                print(f"\nLayer {i} (Pre-feature):")
                print(f"Output shape: {x.shape}")
            
            # Apply domain adaptation at layer 8
            original_features = x
            adapted_features = self.feature_reducer(x)
            x = adapted_features
            features[-1] = adapted_features  # Update the last feature map
            
            print("\nFeature Reduction:")
            print(f"Original shape: {original_features.shape}")
            print(f"Adapted shape: {adapted_features.shape}")
            
            # Post-feature processing (after layer 8)
            for i, layer in enumerate(self.post_feature_layers, start=self.feature_layer + 1):
                # Apply channel adaptation if needed
                x = self._apply_channel_adaptation(x, i)
                
                if isinstance(layer, ultralytics.nn.modules.conv.Concat):
                    x = self._handle_concat_layer(x, layer, features, i)
                elif isinstance(layer, nn.modules.upsampling.Upsample):
                    x = layer(x)
                else:
                    try:
                        x = layer(x)
                    except RuntimeError as e:
                        print(f"\nError at layer {i}:")
                        print(f"Input shape: {x.shape}")
                        if hasattr(layer, 'conv'):
                            print(f"Layer expects {layer.conv.in_channels} input channels")
                        raise
                
                features.append(x)
                print(f"\nLayer {i} (Post-feature):")
                print(f"Output shape: {x.shape}")
            
            return x

def create_domain_adapted_yolo(yolo_model, feature_reducer_weights_path):
    """
    Factory function to create and initialize a domain-adapted YOLO model
    
    Args:
        yolo_model: Base YOLO model
        feature_reducer_weights_path: Path to trained feature reducer weights
    """
    # Initialize domain adapter
    domain_adapter = DomainAdapter(str(yolo_model.weights))
    
    # Load trained weights
    adapter_weights = torch.load(feature_reducer_weights_path)
    domain_adapter.feature_reducer.load_state_dict(adapter_weights['feature_reducer'])
    
    # Create adapted model
    adapted_model = DomainAdaptedYOLO(
        yolo_model=yolo_model,
        feature_reducer=domain_adapter.feature_reducer
    )
    
    # Ensure evaluation mode
    adapted_model.eval()
    for param in adapted_model.parameters():
        param.requires_grad = False
        
    return adapted_model

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

class DebugYOLO(nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.yolo_model = yolo_model.model
        
    def forward(self, x, frame_number=None):
        features = []
        debug_info = {
            'layer_outputs': [],
            'concat_operations': [],
            'feature_maps': []
        }
        
        print(f"\nStarting debug forward pass for frame {frame_number}:")
        print(f"Input shape: {x.shape}")
        
        # Track all feature maps for multi-scale detection
        feature_maps = []
        
        for i, layer in enumerate(self.yolo_model.model):
            layer_info = {
                'layer_idx': i,
                'layer_type': str(type(layer).__name__),
                'input_shape': x.shape
            }
            
            if isinstance(layer, ultralytics.nn.modules.conv.Concat):
                print(f"\nLayer {i} (Concat):")
                print(f"Input shape: {x.shape}")
                print(f"Concat dimension: {layer.d}")
                print(f"Features to concat from: {[f.shape for f in features[-layer.d:]]}")
                
                x = torch.cat([x] + features[-layer.d:], 1)
                print(f"Output shape after concat: {x.shape}")
                
                layer_info['concat_info'] = {
                    'dimension': layer.d,
                    'input_features': [f.shape for f in features[-layer.d:]],
                    'output_shape': x.shape
                }
                
            elif isinstance(layer, ultralytics.nn.modules.block.C3k2):
                print(f"\nLayer {i} (C3k2):")
                print(f"Input shape: {x.shape}")
                x = layer(x)
                print(f"Output shape: {x.shape}")
                
                if i == 8:  # Feature reduction layer
                    print("\n=== FEATURE REDUCER WOULD GO HERE ===")
                    print(f"Input to feature reducer: {x.shape}")
                    feature_maps.append(x.clone())
                    
            elif isinstance(layer, ultralytics.nn.modules.head.Detect):
                print(f"\nLayer {i} (Detect):")
                print(f"Input shape: {x.shape}")
                print(f"Available feature maps: {[fm.shape for fm in feature_maps]}")
                
                detection_features = []
                base_size = feature_maps[-1].shape[-1]
                
                for feat in feature_maps[-3:]:
                    if feat.shape[-1] != base_size:
                        feat = F.interpolate(feat, size=(base_size, base_size), mode='nearest')
                    detection_features.append(feat)
                
                print(f"Detection feature shapes: {[df.shape for df in detection_features]}")
                x = layer(detection_features)
            
            else:
                x = layer(x)
            
            layer_info['output_shape'] = x.shape
            debug_info['layer_outputs'].append(layer_info)
            features.append(x)
            
            # Print detailed shape information for every layer
            print(f"Layer {i} ({layer_info['layer_type']}):")
            print(f"  Input shape:  {layer_info['input_shape']}")
            print(f"  Output shape: {layer_info['output_shape']}")
            print("-" * 50)
        
        return x, debug_info

def debug_frame(model, frame_path, frame_number):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        img = Image.open(frame_path)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Explicitly move tensor to the same device as the model
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Ensure the model is on the same device
        model = model.to(device)
        
        print(f"\nDEBUGGING FRAME {frame_number}")
        print("=" * 50)
        print(f"Input tensor device: {img_tensor.device}")
        print(f"Model device: {next(model.parameters()).device}")
        
        with torch.no_grad():
            output, debug_info = model(img_tensor, frame_number)
        
        return debug_info
        
    except Exception as e:
        print(f"Error during debug of frame {frame_number}: {str(e)}")
        return None

def debug_yolo_architecture(yolo_model):
    """Analyzes the YOLO architecture layer by layer"""
    print("\nYOLO Architecture Analysis:")
    print("=" * 50)
    
    # Get the device of the model
    device = next(yolo_model.model.parameters()).device
    print(f"Model is on device: {device}")
    
    # Create test input tensor on the same device as the model
    x = torch.randn(1, 3, 512, 512).to(device)
    print(f"Input tensor is on device: {x.device}")
    
    features = []
    
    for i, layer in enumerate(yolo_model.model.model):
        # Store original input shape
        input_shape = x.shape
        
        # Layer Info
        print(f"\nLayer {i}: {layer.__class__.__name__}")
        print(f"Input shape: {input_shape}")
        
        # Special layer analysis
        if hasattr(layer, 'conv'):
            w = layer.conv.weight
            print(f"Conv weight shape: {w.shape}")
            print(f"Expected input channels: {layer.conv.in_channels}")
            print(f"Output channels: {layer.conv.out_channels}")
            print(f"Conv weights device: {w.device}")
        
        # Forward pass
        if isinstance(layer, ultralytics.nn.modules.conv.Concat):
            print(f"Concat dimension: {layer.d}")
            print(f"Features to concat: {[f.shape for f in features[-layer.d:]]}")
            x = torch.cat([x] + features[-layer.d:], 1)
        else:
            try:
                x = layer(x)
            except Exception as e:
                print(f"Error in layer {i}: {str(e)}")
                print(f"Input tensor device: {x.device}")
                if hasattr(layer, 'conv'):
                    print(f"Layer weights device: {layer.conv.weight.device}")
                raise
            
        # Store feature
        features.append(x)
        print(f"Output shape: {x.shape}")
        
        # Layer 8 is our target
        if i == 8:
            print("\n=== TARGET LAYER (8) ===")
            print("This is where we inject our feature reducer")
            print("=" * 50)
    
    return features

def xmain():
    """Compare baseline and domain-adapted models on HeiChole dataset"""
    try:
        # Initialize ModelLoader
        loader = ModelLoader()
        
        # Load YOLO model in inference mode and ensure it's frozen
        yolo_model = load_yolo_model(str(loader.yolo_weights))
        yolo_model.model.eval()  # Ensure YOLO is in eval mode
        for param in yolo_model.model.parameters():
            param.requires_grad = False
        
        print("\nAnalyzing YOLO architecture...")
        with torch.no_grad():
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            debug_yolo_architecture(yolo_model)

        print("\nStarting Debug Analysis...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        debug_model = DebugYOLO(yolo_model).to(device)
        #with torch.no_grad():
         #   test_input = torch.randn(1, 3, 512, 512).to(device)
          #  output = debug_model(test_input)
        #print("\nDebug Analysis Complete")

        dataset_dir = str(loader.dataset_path)
        
        frames_to_debug = [
            {'frame': '030300', 'status': 'unknown'},
            {'frame': '030325', 'status': 'unknown'},  # Nehmen Sie einen erfolgreichen Nachbarframe
            {'frame': '030350', 'status': 'unknown'},
            {'frame': '030375', 'status': 'unknown'}
        ]
        
        # Debug jeden Frame
        for frame_info in frames_to_debug:
            frame_number = frame_info['frame']
            frame_path = os.path.join(dataset_dir, "Videos", "VID08", f"{frame_number}.png")
            debug_info = debug_frame(debug_model, frame_path, frame_number)
            
            print(f"\nDebug results for frame {frame_number} ({frame_info['status']}):")
            print("=" * 50)
            if debug_info:
                # Analysieren Sie die Layer-Outputs
                for layer in debug_info['layer_outputs']:
                    if isinstance(layer['layer_type'], ultralytics.nn.modules.conv.Concat):
                        print(f"Concat Layer {layer['layer_idx']}:")
                        print(f"Input shapes: {layer['concat_info']['input_features']}")
                        print(f"Output shape: {layer['concat_info']['output_shape']}")


        # Load domain adapter in eval mode
        domain_adapter = DomainAdapter(str(loader.yolo_weights))
        adapter_weights = torch.load(
            "/home/Bartscht/YOLO/surgical-instrument-action-detection/domain_adaptation/hei_chole/experiments/spatial_domain_adapter_weights_spatial_domain_adaptation/spatial_model_epoch_19.pt"
        )
        
        # Load weights and ensure eval mode
        with torch.no_grad():
            domain_adapter.feature_reducer.load_state_dict(adapter_weights['feature_reducer'])
        domain_adapter.eval()
        
        videos_to_analyze = ["VID08", "VID13"]
        print(f"\nAnalyzing videos: {', '.join(videos_to_analyze)}")
        
        # Create enhanced evaluator
        evaluator = EnhancedHeiCholeEvaluator(
            yolo_model=yolo_model,
            domain_adapter=domain_adapter,
            dataset_dir=dataset_dir
        )
        
        metrics_calculator = BinaryMetricsCalculator(confidence_threshold=CONFIDENCE_THRESHOLD)
        
        predictions = {'baseline': {}, 'adapted': {}}
        ground_truth = {}
        total_frames = 0
        
        # First analyze ground truth distribution
        print("\n==========================================")
        print("GROUND TRUTH ANALYSIS")
        print("==========================================")
        gt_distribution = analyze_label_distribution(dataset_dir, videos_to_analyze)
        
        print("\n==========================================")
        print("MODEL PREDICTIONS ANALYSIS")
        print("==========================================")
        
        # Process each video
        for video in videos_to_analyze:
            # Load ground truth
            gt = evaluator.load_ground_truth(video)
            ground_truth.update(gt)
            
            video_folder = os.path.join(dataset_dir, "Videos", video)
            for frame_file in tqdm(os.listdir(video_folder), desc=f"Processing {video}"):
                if frame_file.endswith('.png'):
                    total_frames += 1
                    frame_id = f"{video}_frame{frame_file.split('.')[0]}"
                    img_path = os.path.join(video_folder, frame_file)
                    
                    # Get predictions from both models
                    frame_predictions = evaluator.evaluate_frame_both_models(
                        img_path,
                        gt[int(frame_file.split('.')[0])],
                        save_visualization=False
                    )
                    
                    if frame_predictions['baseline']:
                        predictions['baseline'][frame_id] = frame_predictions['baseline']
                    if frame_predictions['adapted']:
                        predictions['adapted'][frame_id] = frame_predictions['adapted']
        
        print(f"\nProcessed {total_frames} frames in total.")
        
        # Calculate and compare metrics
        print("\n====== MODEL COMPARISON REPORT ======")
        for model_type in ['baseline', 'adapted']:
            print(f"\n{model_type.upper()} MODEL RESULTS:")
            print("=" * 50)
            metrics = metrics_calculator.calculate_metrics(
                predictions[model_type],
                ground_truth
            )
            print_metrics_report(metrics, total_frames)
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
def main():
    """Test domain-adapted YOLO model with feature reducer"""
    try:
        print("\n=== Initializing Models ===")
        # Initialize ModelLoader
        loader = ModelLoader()
        
        # Load base YOLO model
        print("\nLoading base YOLO model...")
        yolo_model = load_yolo_model(str(loader.yolo_weights))
        yolo_model.model.eval()
        
        # Path to feature reducer weights
        weights_path = "/home/Bartscht/YOLO/surgical-instrument-action-detection/domain_adaptation/hei_chole/experiments/spatial_domain_adapter_weights_spatial_domain_adaptation/spatial_model_epoch_19.pt"
        
        print("\n=== Creating Domain-Adapted YOLO ===")
        # Initialize domain adapter with YOLO path
        domain_adapter = DomainAdapter(str(loader.yolo_weights))  # Hier ist die Korrektur
        
        # Load feature reducer weights
        print(f"\nLoading feature reducer weights from: {weights_path}")
        checkpoint = torch.load(weights_path)
        domain_adapter.feature_reducer.load_state_dict(checkpoint['feature_reducer'])
        
        # Create adapted model
        adapted_yolo = DomainAdaptedYOLO(
            yolo_model=yolo_model,
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
                    print(f"Successfully processed frame {frame_number}")
                    print(f"Output shape: {predictions.shape if hasattr(predictions, 'shape') else 'N/A'}")
                    
            except Exception as e:
                print(f"Error processing frame {frame_number}: {str(e)}")
                traceback.print_exc()
                continue
        
        print("\n=== Testing Complete ===")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        
if __name__ == '__main__':
    main()