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
        self.feature_layer = 16  # Angepasst auf Layer 16
        
        for param in self.yolo_model.parameters():
            param.requires_grad = False
            
        # Feature Reducer für Layer 16 (256 Kanäle)
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256, 256),
            nn.Conv2d(256, 256, 
                     kernel_size=3, 
                     padding=1, 
                     groups=32),
            nn.BatchNorm2d(256),
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


class xDomainAdaptedYOLO(nn.Module):
    def __init__(self, yolo_model, feature_reducer):
        super().__init__()
        self.yolo_model = yolo_model.model
        self.feature_layer = 16
        self.feature_reducer = feature_reducer

        # Channel configuration based on YOLO architecture
        self.channel_config = {
            19: 768,  # C3k2 block expects 768 input channels
            22: 1024,  # Final C3k2 block
            23: {  # Detect layer expects specific channels for each feature level
                16: 256,  # P3/8-small
                19: 512,  # P4/16-medium
                22: 1024  # P5/32-large
            }
        }
        # Store output features for detection
        self.detection_features = {}

        # Split network into sections
        self.pre_feature_layers = nn.ModuleList()
        self.feature_layer_16 = None
        self.post_feature_layers = nn.ModuleList()

        # Track skip connection structure
        self.skip_connections = self._analyze_skip_connections()

        # Split model into sections
        for i, layer in enumerate(self.yolo_model.model):
            if i < self.feature_layer:
                self.pre_feature_layers.append(layer)
            elif i == self.feature_layer:
                self.feature_layer_16 = layer
            else:
                self.post_feature_layers.append(layer)

        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def _analyze_skip_connections(self):
        """Analyze and map all skip connections in the network"""
        skip_connections = {}
        features_needed = set()

        for i, layer in enumerate(self.yolo_model.model):
            if isinstance(layer, ultralytics.nn.modules.conv.Concat):
                sources = list(range(i - layer.d, i))
                skip_connections[i] = sources
                features_needed.update(sources)

        return skip_connections

    def _get_target_channels(self, layer_idx):
        """Determine target number of channels for a given layer"""
        return self.channel_config.get(layer_idx, None)

    def _adjust_channels(self, x, target_channels):
        """Adjust number of channels using 1x1 convolution if needed"""
        if target_channels is None or x.shape[1] == target_channels:
            return x

        adapter = nn.Conv2d(x.shape[1], target_channels, kernel_size=1).to(x.device)
        return adapter(x)

    def _handle_concat(self, x, features, layer_idx, layer):
        """Handle concat operations with proper feature management"""
        print(f"\nConcat at layer {layer_idx}:")
        print(f"Current feature shape: {x.shape}")

        # Determine target output channels based on network architecture
        next_layer_idx = layer_idx + 1
        target_channels = self._get_target_channels(next_layer_idx)

        concat_features = [x]  # Start with current feature map

        # Get source layers for this concat operation
        if layer_idx in self.skip_connections:
            source_layers = self.skip_connections[layer_idx]

            for source_idx in source_layers:
                if source_idx < len(features):
                    skip_feature = features[source_idx]
                    print(f"Adding skip connection from layer {source_idx}, shape: {skip_feature.shape}")

                    # Handle spatial dimension mismatches if needed
                    if skip_feature.shape[-2:] != x.shape[-2:]:
                        skip_feature = F.interpolate(
                            skip_feature, 
                            size=x.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )

                    concat_features.append(skip_feature)
                else:
                    print(f"Warning: Skip connection source layer {source_idx} not available")

        # Perform concatenation
        x = torch.cat(concat_features, dim=1)
        print(f"After concat shape: {x.shape}")

        # Adjust channels if needed for next layer
        if target_channels is not None:
            x = self._adjust_channels(x, target_channels)
            print(f"After channel adjustment: {x.shape}")

        return x

    def forward(self, x):
        features = []
        detection_features = {}

        print("\nStarting forward pass")
        print(f"Input shape: {x.shape}")

        # Pre-feature extraction (before layer 16)
        for i, layer in enumerate(self.pre_feature_layers):
            if isinstance(layer, ultralytics.nn.modules.conv.Concat):
                x = self._handle_concat(x, features, i, layer)
            else:
                print(f"\nLayer {i}: {layer.__class__.__name__}")
                print(f"Input shape: {x.shape}")
                x = layer(x)
                print(f"Output shape: {x.shape}")
            features.append(x)

        # Layer 16 processing
        print("\nProcessing Layer 16:")
        print(f"Input shape: {x.shape}")
        x = self.feature_layer_16(x)
        print(f"Layer 16 output shape: {x.shape}")
        original_features = x.clone()
        features.append(original_features)

        # Apply domain-invariant feature reducer
        print("\nApplying feature reducer:")
        x = self.feature_reducer(x)
        print(f"Feature reducer output shape: {x.shape}")

        # Ensure Layer 16 feature map is ready for detection
        x = self._adjust_channels(x, target_channels=self.channel_config[23][16])
        detection_features[16] = x

        # Post-feature processing
        for i, layer in enumerate(self.post_feature_layers):
            layer_idx = i + self.feature_layer + 1

            if isinstance(layer, ultralytics.nn.modules.conv.Concat):
                x = self._handle_concat(x, features, layer_idx, layer)
            else:
                print(f"\nLayer {layer_idx}: {layer.__class__.__name__}")
                print(f"Input shape: {x.shape}")
                x = layer(x)
                print(f"Output shape: {x.shape}")

            features.append(x)

            # Store and adjust features for detection
            if layer_idx == 19:
                x = self._adjust_channels(x, target_channels=self.channel_config[23][19])
                detection_features[19] = x
            elif layer_idx == 22:
                x = self._adjust_channels(x, target_channels=self.channel_config[23][22])
                detection_features[22] = x

        # Prepare for Detect layer
        detect_layer = self.yolo_model.model[-1]
        if isinstance(detect_layer, ultralytics.nn.modules.head.Detect):
            return detect_layer([
                detection_features[16],  # P3/8-small
                detection_features[19],  # P4/16-medium
                detection_features[22]   # P5/32-large
            ])

        return x


class yDomainAdaptedYOLO(nn.Module):
    def __init__(self, yolo_model, feature_reducer):
        super().__init__()
        self.yolo_model = yolo_model
        self.feature_layer = 16
        self.feature_reducer = feature_reducer

        # Channel configuration basierend auf YOLO Architektur
        self.channel_config = {
            19: 768,  # C3k2 block erwartet 768 input channels
            22: 1024,  # Final C3k2 block
            23: {  # Detect layer erwartet spezifische channels für jedes feature level
                16: 256,  # P3/8-small
                19: 512,  # P4/16-medium
                22: 1024  # P5/32-large
            }
        }

        # Teile das Modell in Sektionen
        self.pre_feature_layers = nn.ModuleList()
        self.feature_layer_16 = None
        self.post_feature_layers = nn.ModuleList()

        for i, layer in enumerate(self.yolo_model.model.model):
            if i < self.feature_layer:
                self.pre_feature_layers.append(layer)
            elif i == self.feature_layer:
                self.feature_layer_16 = layer
            else:
                self.post_feature_layers.append(layer)

    def _handle_concat(self, x, features, layer_idx, layer):
        print(f"\nConcat at layer {layer_idx}:")
        print(f"Current feature shape: {x.shape}")

        # Bestimme target output channels basierend auf Netzwerk-Architektur
        next_layer_idx = layer_idx + 1
        target_channels = self._get_target_channels(next_layer_idx)

        # Start mit current feature map
        concat_features = [x]

        # Hole source layers für diese concat operation
        source_indices = list(range(len(features)-layer.d, len(features)))
        for source_idx in source_indices:
            if source_idx < len(features):
                skip_feature = features[source_idx]
                print(f"Adding skip connection from layer {source_idx}, shape: {skip_feature.shape}")
                
                # Handle spatial dimension mismatches wenn nötig
                if skip_feature.shape[-2:] != x.shape[-2:]:
                    skip_feature = F.interpolate(
                        skip_feature, 
                        size=x.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                concat_features.append(skip_feature)

        # Perform concatenation
        x = torch.cat(concat_features, dim=1)
        print(f"After concat shape: {x.shape}")

        # Adjust channels if needed for next layer
        if target_channels is not None:
            x = self._adjust_channels(x, target_channels)
            print(f"After channel adjustment: {x.shape}")

        return x

    def _get_target_channels(self, layer_idx):
        """Bestimme target number of channels für einen gegebenen layer"""
        return self.channel_config.get(layer_idx, None)

    def _adjust_channels(self, x, target_channels):
        """Adjust number of channels using 1x1 convolution if needed"""
        if x.shape[1] != target_channels:
            adapter = nn.Conv2d(x.shape[1], target_channels, kernel_size=1).to(x.device)
            return adapter(x)
        return x

    def forward(self, x):
        features = []
        detection_features = {}

        print("\nStarting forward pass")
        print(f"Input shape: {x.shape}")

        # Pre-feature extraction (before layer 16)
        for i, layer in enumerate(self.pre_feature_layers):
            if isinstance(layer, ultralytics.nn.modules.conv.Concat):
                x = self._handle_concat(x, features, i, layer)
            else:
                print(f"\nLayer {i}: {layer.__class__.__name__}")
                print(f"Input shape: {x.shape}")
                x = layer(x)
                print(f"Output shape: {x.shape}")
            features.append(x)

        # Layer 16 processing (P3)
        print("\nProcessing Layer 16 (P3):")
        print(f"Input shape: {x.shape}")
        x = self.feature_layer_16(x)
        print(f"Layer 16 output shape: {x.shape}")
        original_features = x.clone()
        features.append(original_features)

        # Apply domain-invariant feature reducer
        print("\nApplying feature reducer:")
        x = self.feature_reducer(x)
        print(f"Feature reducer output shape: {x.shape}")

        # Store P3 (Layer 16) - Should be 256 channels
        detection_features[16] = self._adjust_channels(x, target_channels=256)

        # Post-feature processing
        for i, layer in enumerate(self.post_feature_layers):
            layer_idx = i + self.feature_layer + 1

            if isinstance(layer, ultralytics.nn.modules.conv.Concat):
                x = self._handle_concat(x, features, layer_idx, layer)
            else:
                print(f"\nLayer {layer_idx}: {layer.__class__.__name__}")
                print(f"Input shape: {x.shape}")
                x = layer(x)
                print(f"Output shape: {x.shape}")

            features.append(x)

            # Store P4 (Layer 19) - Should be 512 channels
            if layer_idx == 19:
                detection_features[19] = self._adjust_channels(x, target_channels=512)
            # Store P5 (Layer 22) - Should be 1024 channels
            elif layer_idx == 22:
                detection_features[22] = self._adjust_channels(x, target_channels=1024)

        # Prepare for Detect layer
        detect_layer = self.yolo_model.model.model[-1]
        processed_features = []

        # Process each feature map with its corresponding cv2 and cv3
        for i, layer_idx in enumerate([16, 19, 22]):  # P3, P4, P5
            feat = detection_features[layer_idx]
            print(f"\nProcessing detection feature from layer {layer_idx}:")
            print(f"Input shape: {feat.shape}")
            
            # Apply bbox head (cv2) and class head (cv3)
            bbox_feat = detect_layer.cv2[i](feat)
            cls_feat = detect_layer.cv3[i](feat)
            print(f"bbox_feat shape: {bbox_feat.shape}")
            print(f"cls_feat shape: {cls_feat.shape}")
            
            # Concatenate bbox and class features
            processed = torch.cat([bbox_feat, cls_feat], 1)
            print(f"Processed feature shape: {processed.shape}")
            processed_features.append(processed)

        # Final detection
        print("\nPassing to final Detect layer")
        return detect_layer(processed_features)

class zDomainAdaptedYOLO(nn.Module):
    def __init__(self, yolo_model, feature_reducer):
        super().__init__()
        self.yolo_model = yolo_model.model
        self.feature_layer = 16
        self.feature_reducer = feature_reducer

        self.channel_config = {
            19: 768,
            22: 1024,
            23: {
                16: 256,
                19: 512,
                22: 1024
            }
        }
        self.detection_features = {}
        self.pre_feature_layers = nn.ModuleList()
        self.feature_layer_16 = None
        self.post_feature_layers = nn.ModuleList()
        self.skip_connections = self._analyze_skip_connections()

        for i, layer in enumerate(self.yolo_model.model):
            if i < self.feature_layer:
                self.pre_feature_layers.append(layer)
            elif i == self.feature_layer:
                self.feature_layer_16 = layer
            else:
                self.post_feature_layers.append(layer)

        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def _analyze_skip_connections(self):
        skip_connections = {}
        features_needed = set()
        for i, layer in enumerate(self.yolo_model.model):
            if isinstance(layer, ultralytics.nn.modules.conv.Concat):
                sources = list(range(i - layer.d, i))
                skip_connections[i] = sources
                features_needed.update(sources)
        return skip_connections

    def _get_target_channels(self, layer_idx):
        return self.channel_config.get(layer_idx, None)

    def _adjust_channels(self, x, target_channels):
        if target_channels is None or x.shape[1] == target_channels:
            return x
        adapter = nn.Conv2d(x.shape[1], target_channels, kernel_size=1).to(x.device)
        return adapter(x)

    def _handle_concat(self, x, features, layer_idx, layer):
        print(f"\nConcat at layer {layer_idx}:")
        print(f"Current feature shape: {x.shape}")

        next_layer_idx = layer_idx + 1
        target_channels = self._get_target_channels(next_layer_idx)
        concat_features = [x]

        if layer_idx in self.skip_connections:
            source_layers = self.skip_connections[layer_idx]
            for source_idx in source_layers:
                if source_idx < len(features):
                    skip_feature = features[source_idx]
                    print(f"Adding skip connection from layer {source_idx}, shape: {skip_feature.shape}")
                    if skip_feature.shape[-2:] != x.shape[-2:]:
                        skip_feature = F.interpolate(skip_feature, size=x.shape[-2:],
                                                   mode='bilinear', align_corners=False)
                    concat_features.append(skip_feature)

        x = torch.cat(concat_features, dim=1)
        print(f"After concat shape: {x.shape}")

        if target_channels is not None:
            x = self._adjust_channels(x, target_channels)
            print(f"After channel adjustment: {x.shape}")
        return x

    def forward(self, x):
        features = []
        detection_features = {}

        print("\nStarting forward pass")
        print(f"Input shape: {x.shape}")

        # Pre-feature extraction (before layer 16)
        for i, layer in enumerate(self.pre_feature_layers):
            if isinstance(layer, ultralytics.nn.modules.conv.Concat):
                x = self._handle_concat(x, features, i, layer)
            else:
                print(f"\nLayer {i}: {layer.__class__.__name__}")
                print(f"Input shape: {x.shape}")
                x = layer(x)
                print(f"Output shape: {x.shape}")
            features.append(x)

        # Layer 16 processing - Original YOLO Layer
        print("\nProcessing Layer 16:")
        print(f"Input shape: {x.shape}")
        x = self.feature_layer_16(x)
        print(f"Layer 16 output shape: {x.shape}")
        
        # Feature Reducer nach Layer 16
        print("\nApplying feature reducer:")
        x = self.feature_reducer(x)
        print(f"Feature reducer output shape: {x.shape}")
        
        # Wichtig: Wir speichern die domäneninvarianten Features für skip connections
        features.append(x)
        detection_features[16] = x

        # Post-feature processing
        for i, layer in enumerate(self.post_feature_layers):
            layer_idx = i + self.feature_layer + 1

            if isinstance(layer, ultralytics.nn.modules.conv.Concat):
                x = self._handle_concat(x, features, layer_idx, layer)
            else:
                print(f"\nLayer {layer_idx}: {layer.__class__.__name__}")
                print(f"Input shape: {x.shape}")
                x = layer(x)
                print(f"Output shape: {x.shape}")

            features.append(x)

            if layer_idx == 19:
                detection_features[19] = self._adjust_channels(x, self.channel_config[23][19])
            elif layer_idx == 22:
                detection_features[22] = self._adjust_channels(x, self.channel_config[23][22])

        detect_layer = self.yolo_model.model[-1]
        if isinstance(detect_layer, ultralytics.nn.modules.head.Detect):
            return detect_layer([
                detection_features[16],
                detection_features[19],
                detection_features[22]
            ])

        return x

class DomainAdaptedYOLO(nn.Module):
    def __init__(self, yolo_model, feature_reducer):
        super().__init__()
        self.yolo_model = yolo_model.model
        self.feature_reducer = feature_reducer
        
        # Registriere Forward-Hook nach Layer 16
        self.layer_16 = None
        for i, layer in enumerate(self.yolo_model.model):
            if i == 16:
                # Register the hook to execute AFTER layer 16 completes
                layer.register_forward_hook(self._post_layer_16_hook)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def _post_layer_16_hook(self, module, input_feat, output_feat):
        """
        Hook der NACH Layer 16 ausgeführt wird
        - output_feat enthält den kompletten Output des C3k2 Blocks
        """
        print(f"\nFeature Hook aktiviert:")
        print(f"Input shape: {[f.shape for f in input_feat]}")
        print(f"Output shape vor Feature Reducer: {output_feat.shape}")
        
        # Wende Feature Reducer an
        modified = self.feature_reducer(output_feat)
        print(f"Output shape nach Feature Reducer: {modified.shape}")
        
        return modified
    
    def forward(self, x):
        """Normaler Forward-Pass - der Hook modifiziert automatisch die Features"""
        return self.yolo_model.model(x)
    
def _prepare_feature_for_detect(self, feature, cv2_layer, cv3_layer):
    """
    Prepare feature map to match Detect layer's convolutional layer expectations
    
    Args:
        feature (torch.Tensor): Input feature map
        cv2_layer (nn.Sequential): Bounding box regression convolution layers
        cv3_layer (nn.Sequential): Classification convolution layers
    
    Returns:
        torch.Tensor: Prepared feature map
    """
    # Get the expected input channel for the first conv layer in cv2
    expected_channels = cv2_layer[0].conv.in_channels
    
    # Adjust channels if needed
    if feature.shape[1] != expected_channels:
        channel_adapter = nn.Conv2d(feature.shape[1], expected_channels, kernel_size=1).to(feature.device)
        feature = channel_adapter(feature)
    
    return feature

# Helper function to ensure channel dimensions match
def adjust_channel_dimensions(x, target_channels):
    """Adjust number of channels using 1x1 convolutions if needed"""
    if x.shape[1] != target_channels:
        adapter = nn.Conv2d(x.shape[1], target_channels, kernel_size=1).to(x.device)
        return adapter(x)
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

class YOLODebugger(nn.Module):
   def __init__(self, yolo_model):
       super().__init__()
       self.yolo_model = yolo_model.model

   def forward(self, x):
       features = []
       skip_connections = []
       
       print("\n=== Feature Flow Analysis ===")
       
       for i, layer in enumerate(self.yolo_model.model):
           print(f"\nLayer {i}: {layer.__class__.__name__}")
           print(f"Input shape: {x.shape}")
           
           if isinstance(layer, ultralytics.nn.modules.conv.Concat):
               print("\nCONCAT OPERATION:")
               print(f"Current features: {x.shape}")
               for j, feat in enumerate(features[-layer.d:]):
                   print(f"Concat feature {j}: {feat.shape}")
                   origin_idx = len(features) - layer.d + j
                   print(f"From layer: {origin_idx}")
               skip_connections.append({
                   'layer': i,
                   'sources': [len(features) - layer.d + j for j in range(layer.d)]
               })
               x = torch.cat([x] + features[-layer.d:], 1)
               print(f"After concat: {x.shape}")
           
           else:
               x = layer(x)
           
           features.append(x)
           
           # Spezielle Analyse für Layer 16-19
           if 15 <= i <= 19:
               print(f"\nDETAILED ANALYSIS Layer {i}:")
               if isinstance(layer, ultralytics.nn.modules.block.C3k2):
                   print("C3k2 internals:")
                   if hasattr(layer, 'cv1'):
                       print(f"cv1 weight shape: {layer.cv1.conv.weight.shape}")
               print(f"Output Features: {x.shape}")
       
       print("\n=== Skip Connection Summary ===")
       for conn in skip_connections:
           print(f"\nLayer {conn['layer']} receives from layers: {conn['sources']}")
       
       return x

def run_debug_analysis(img_path, yolo_model):
   debugger = YOLODebugger(yolo_model).cuda()
   img = Image.open(img_path)
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   img_tensor = transform(img).unsqueeze(0).cuda()
   
   with torch.no_grad():
       debugger(img_tensor)


class DebugDomainYOLO(nn.Module):
    def __init__(self, yolo_model, feature_reducer):
        super().__init__()
        self.yolo_model = yolo_model.model
        self.feature_layer = 16
        self.feature_reducer = feature_reducer

    def forward(self, x):
        features = []
        print("\nStarting forward pass:")
        print(f"Input shape: {x.shape}")
        
        for i, layer in enumerate(self.yolo_model.model):
            print(f"\nLayer {i}: {layer.__class__.__name__}")
            print(f"Input shape: {x.shape}")
            
            if isinstance(layer, ultralytics.nn.modules.conv.Concat):
                print("Concat operation details:")
                print(f"Features to concat: {[f.shape for f in features[-layer.d:]]}")
                x = torch.cat([x] + features[-layer.d:], 1)
            else:
                x = layer(x)
            
            features.append(x)
            print(f"Output shape: {x.shape}")
            
            # Nach Layer 16 Feature-Reducer anwenden
            if i == self.feature_layer:
                print("\nApplying feature reducer after Layer 16:")
                original_shape = x.shape
                x = self.feature_reducer(x)
                print(f"Shape change: {original_shape} -> {x.shape}")
                features[-1] = x
                
        return x
    
# Nutzung:
def debug_run(img_path, yolo_model, feature_reducer):
    debug_model = DebugDomainYOLO(yolo_model, feature_reducer).cuda()
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).cuda()
    
    with torch.no_grad():
        debug_model(img_tensor)


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
        weights_path = "/home/Bartscht/YOLO/surgical-instrument-action-detection/domain_adaptation/hei_chole/experiments/spatial_domain_adapter_weights_spatial_domain_adaptation/spatial_model_epoch_2.pt"
        
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
        
        

        #print("\nStarting debug run...")
        #test_frame_path = os.path.join(loader.dataset_path, "Videos", "VID08", "030300.png")
        #run_debug_analysis(test_frame_path, yolo_model)
        #debug_run(test_frame_path, yolo_model, domain_adapter.feature_reducer)

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