import os
import sys
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
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
        
        # Feature Reducer
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256, 256),
            nn.Conv2d(256, 256, 3, padding=1, groups=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Note: AdaptiveAvgPool2d and Flatten are only used during training
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
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

class DomainAdaptedYOLO(nn.Module):
    def __init__(self, yolo_model, feature_reducer):
        super().__init__()
        self.yolo_model = yolo_model.model
        self.feature_layer = 8
        
        # Split YOLO layers
        self.pre_feature_layers = nn.ModuleList([
            layer for i, layer in enumerate(self.yolo_model.model) 
            if i <= self.feature_layer
        ])
        self.post_feature_layers = nn.ModuleList([
            layer for i, layer in enumerate(self.yolo_model.model) 
            if i > self.feature_layer
        ])
        
        # Modified feature reducer for inference (removing pooling/flatten)
        self.feature_reducer = self._modify_feature_reducer(feature_reducer)
        
        # Ensure evaluation mode
        self._freeze_all_layers()
    
    def _modify_feature_reducer(self, original_reducer):
        """Creates inference version of feature reducer without pooling/flatten"""
        # Create new sequential without final pooling/flatten
        layers = []
        for layer in original_reducer:
            if not isinstance(layer, (nn.AdaptiveAvgPool2d, nn.Flatten)):
                layers.append(layer)
        
        # Add final conv to match YOLO's expected channels
        layers.append(nn.Conv2d(256, 512, 1))  # Restore channel dimension to 512
        
        # Create new sequential model
        inference_reducer = nn.Sequential(*layers)
        
        # Copy weights from original layers
        with torch.no_grad():
            for i, layer in enumerate(layers[:-1]):  # Exclude final conv
                if hasattr(layer, 'weight'):
                    layer.weight.copy_(original_reducer[i].weight)
                if hasattr(layer, 'bias'):
                    layer.bias.copy_(original_reducer[i].bias)
        
        return inference_reducer
    
    def _freeze_all_layers(self):
        """Ensure all layers are frozen and in eval mode"""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        with torch.no_grad():
            # Pre-feature extraction
            features = x
            for layer in self.pre_feature_layers:
                features = layer(features)
            
            # Apply modified feature reducer (maintains spatial dimensions)
            reduced_features = self.feature_reducer(features)
            
            # Post-feature processing
            x = reduced_features
            for layer in self.post_feature_layers:
                x = layer(x)
            
            return x

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
        super().__init__(yolo_model, dataset_dir)
        
        # Ensure domain adapter is in eval mode
        domain_adapter.eval()
        for param in domain_adapter.parameters():
            param.requires_grad = False
        
        # Create adapted YOLO model and ensure it's in eval mode
        self.adapted_model = DomainAdaptedYOLO(
            yolo_model,
            domain_adapter.feature_reducer
        ).to(self.device)
        
        # Double-check everything is in eval mode
        self.adapted_model.eval()
        for param in self.adapted_model.parameters():
            param.requires_grad = False

    def evaluate_frame_both_models(self, img_path, ground_truth, save_visualization=True):
        """Evaluates a frame with both baseline and domain-adapted models"""
        frame_number = int(os.path.basename(img_path).split('.')[0])
        video_name = os.path.basename(os.path.dirname(img_path))
        
        # Get baseline predictions
        baseline_predictions = self.evaluate_frame(img_path, ground_truth, save_visualization)
        
        # Process with adapted model
        adapted_predictions = []
        try:
            img = Image.open(img_path)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            # Get predictions from adapted model - ensure no gradients
            with torch.no_grad():
                # Forward pass through adapted model
                adapted_output = self.adapted_model(img_tensor)
                
                # Process detections
                for box in adapted_output[0].boxes:
                    instrument_class = int(box.cls)
                    confidence = float(box.conf)
                    
                    if confidence >= CONFIDENCE_THRESHOLD:
                        if instrument_class in IGNORED_INSTRUMENTS:
                            continue
                            
                        try:
                            cholect50_instrument = TOOL_MAPPING[instrument_class]
                        except KeyError:
                            continue
                        
                        mapped_instrument = CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING.get(cholect50_instrument)
                        
                        if mapped_instrument:
                            adapted_predictions.append({
                                'frame_id': f"{video_name}_frame{frame_number}",
                                'instrument': {
                                    'name': mapped_instrument,
                                    'confidence': confidence,
                                    'binary_pred': 1 if confidence >= CONFIDENCE_THRESHOLD else 0
                                }
                            })
                    
        except Exception as e:
            print(f"Error processing adapted model for frame {frame_number}: {str(e)}")
        
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

def main():
    """Compare baseline and domain-adapted models on HeiChole dataset"""
    try:
        # Initialize ModelLoader
        loader = ModelLoader()
        
        # Load YOLO model in inference mode and ensure it's frozen
        yolo_model = load_yolo_model(str(loader.yolo_weights))
        yolo_model.model.eval()  # Ensure YOLO is in eval mode
        for param in yolo_model.model.parameters():
            param.requires_grad = False
            
        dataset_dir = str(loader.dataset_path)
        
        # Load domain adapter in eval mode
        domain_adapter = DomainAdapter(str(loader.yolo_weights))
        adapter_weights = torch.load(
            "/home/Bartscht/YOLO/surgical-instrument-action-detection/domain_adaptation/hei_chole/experiments/domain_adapter_weights/model_epoch_2.pt"
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

if __name__ == '__main__':
    main()