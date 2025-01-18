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
current_dir = Path(__file__).resolve().parent  # GrasP/evaluation
hei_chole_dir = current_dir.parent  # GrasP
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
IOU_THRESHOLD = 0.5

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

# Mapping between CholecT50 and GraSP instruments
CHOLECT50_TO_GRASP_INSTRUMENT_MAPPING = {
    'grasper': ['Prograsp Forceps', 'Laparoscopic Grasper'],
    'bipolar': ['Bipolar Forceps'],
    'scissors': ['Monopolar Curved Scissors'],
    'clipper': ['Clip Applier'],
    'irrigator': ['Suction Instrument'],
    'hook': None  # No corresponding instrument in GraSP
}

# Reverse mapping for evaluation
GRASP_TO_CHOLECT50_INSTRUMENT_MAPPING = {
    'Prograsp Forceps': 'grasper',
    'Laparoscopic Grasper': 'grasper',
    'Bipolar Forceps': 'bipolar',
    'Monopolar Curved Scissors': 'scissors',
    'Clip Applier': 'clipper',
    'Suction Instrument': 'irrigator',
    'Large Needle Driver': None  # No corresponding instrument in CholecT50
}

CHOLECT50_TO_HEICHOLE_VERB_MAPPING = {
    'grasp': ['Hold', 'Still', 'Release'],
    'retract': ['Pull', 'Still'],
    'dissect': None,
    'coagulate': ['Cauterize', 'Hold'],
    'clip': ['Close'],
    'cut': ['Cut'],
    'aspirate': ['Suction'],
    'irrigate': ['Suction'],
    'pack': ['Push', 'Other'],
    'null_verb': ['Travel', 'Push', 'Open']
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
        
        # Dataset path for GraSP
        self.dataset_path = Path("/data/Bartscht/GrasP/test")
        self.videos_path = self.dataset_path / "Videos"
        
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
        

class GraSPEvaluator:
    def __init__(self, yolo_model, verb_model, dataset_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = yolo_model
        self.verb_model = verb_model.to(self.device)
        self.verb_model.eval()
        self.dataset_dir = Path(dataset_dir)
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Mapping dictionaries from GTLoader
        self.instrument_id_to_name = {
            1: 'Bipolar Forceps',
            2: 'Prograsp Forceps',
            3: 'Large Needle Driver',
            4: 'Monopolar Curved Scissors',
            5: 'Suction Instrument',
            6: 'Clip Applier',
            7: 'Laparoscopic Grasper'
        }
        
        self.action_id_to_name = {
            1: 'Cauterize', 2: 'Close', 3: 'Cut',
            4: 'Grasp', 5: 'Hold', 6: 'Open',
            7: 'Open Something', 8: 'Pull', 9: 'Push',
            10: 'Release', 11: 'Still', 12: 'Suction',
            13: 'Travel', 14: 'Other'
        }
        
        self.reset_counters()

    def reset_counters(self):
        """Reset all counters and data structures"""
        # Ground truth counters
        self.gt_instrument_counts = {name: 0 for name in self.instrument_id_to_name.values()}
        self.gt_action_counts = {name: 0 for name in self.action_id_to_name.values()}
        self.gt_instrument_action_pairs = defaultdict(int)
        
        # Prediction counters
        self.pred_instrument_counts = {name: 0 for name in self.instrument_id_to_name.values()}
        self.pred_action_counts = {name: 0 for name in self.action_id_to_name.values()}
        self.pred_instrument_action_pairs = defaultdict(int)
        
        # Frame-wise data storage
        self.frame_data = defaultdict(lambda: {
            'gt_instruments': [],    # Ground truth instruments
            'gt_actions': [],        # Ground truth actions
            'gt_pairs': [],          # Ground truth instrument-action pairs
            'pred_instruments': [],   # Predicted instruments
            'pred_actions': [],      # Predicted actions
            'pred_pairs': []         # Predicted instrument-action pairs
        })

    def load_ground_truth(self, video_id):
        """Load ground truth annotations for a video"""
        json_file = self.dataset_dir / "Labels" / f"{video_id}.json"
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            print(f"\nLoading annotations from: {json_file}")
            self.reset_counters()
            
            # Process each frame
            for frame_id, frame_data in data['frames'].items():
                frame_num = int(frame_id.split('.')[0])
                frame_results = self.frame_data[frame_num]
                
                # Process each instrument
                for instrument in frame_data.get('instruments', []):
                    instance_id = instrument.get('id')
                    category_id = instrument.get('category_id')
                    
                    if category_id in self.instrument_id_to_name:
                        instrument_name = self.instrument_id_to_name[category_id]
                        
                        # Count ground truth instrument
                        self.gt_instrument_counts[instrument_name] += 1
                        
                        # Store instrument info
                        instrument_info = {
                            'instance_id': instance_id,
                            'name': instrument_name,
                            'bbox': instrument.get('bbox', []),
                            'category_id': category_id
                        }
                        
                        # Process actions for this instrument
                        actions = instrument.get('actions', [])
                        if isinstance(actions, list):
                            valid_actions = []
                            for action_id in actions:
                                if action_id in self.action_id_to_name:
                                    action_name = self.action_id_to_name[action_id]
                                    valid_actions.append(action_name)
                                    
                                    # Count ground truth action
                                    self.gt_action_counts[action_name] += 1/len(actions)
                                    
                                    # Count ground truth pair
                                    pair_key = f"{instrument_name}_{action_name}"
                                    self.gt_instrument_action_pairs[pair_key] += 1/len(actions)
                            
                            if valid_actions:
                                instrument_info['valid_actions'] = valid_actions
                        
                        frame_results['gt_instruments'].append(instrument_info)
                        
                        for action in valid_actions:
                            frame_results['gt_pairs'].append({
                                'instrument_id': instance_id,
                                'instrument_name': instrument_name,
                                'action': action
                            })
            
            return True
            
        except Exception as e:
            print(f"Error loading ground truth: {str(e)}")
            return False

    def evaluate_frame(self, img_path, frame_annotations, save_visualization=False):
        """Evaluate a single frame with overlap-based verb validation"""
        frame_predictions = []
        frame_number = int(os.path.basename(img_path).split('.')[0])
        
        try:
            # Load and process image
            img = Image.open(img_path)
            yolo_results = self.yolo_model(img)
            
            # Process detections
            for detection in yolo_results[0].boxes:
                instrument_class = int(detection.cls)
                confidence = float(detection.conf)
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    cholect50_instrument = TOOL_MAPPING[instrument_class]
                    grasp_instruments = CHOLECT50_TO_GRASP_INSTRUMENT_MAPPING.get(cholect50_instrument)
                    
                    if grasp_instruments:
                        if isinstance(grasp_instruments, str):
                            grasp_instruments = [grasp_instruments]
                            
                        for grasp_instrument in grasp_instruments:
                            # Update instrument predictions
                            self.pred_instrument_counts[grasp_instrument] += 1
                            
                            # Get verb predictions
                            box = detection.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, box)
                            instrument_crop = img.crop((x1, y1, x2, y2))
                            crop_tensor = self.transform(instrument_crop).unsqueeze(0).to(self.device)
                            
                            verb_outputs = self.verb_model(crop_tensor, [cholect50_instrument])
                            verb_probs = verb_outputs['probabilities']
                            
                            # Get ground truth actions for this instrument in this frame
                            gt_actions = set()
                            for gt_inst in frame_annotations['gt_instruments']:
                                if gt_inst['name'] == grasp_instrument and 'valid_actions' in gt_inst:
                                    gt_actions.update(gt_inst['valid_actions'])
                            
                            # Process verb predictions
                            for verb_idx in torch.topk(verb_probs[0], k=1).indices.cpu().numpy():
                                cholect50_verb = VERB_MAPPING[VERB_MODEL_TO_EVAL_MAPPING[verb_idx]]
                                possible_grasp_verbs = CHOLECT50_TO_HEICHOLE_VERB_MAPPING.get(cholect50_verb, [])
                                verb_confidence = float(verb_probs[0][verb_idx])
                                
                                if possible_grasp_verbs:
                                    # Convert to set if it's a single string
                                    if isinstance(possible_grasp_verbs, str):
                                        possible_grasp_verbs = {possible_grasp_verbs}
                                    else:
                                        possible_grasp_verbs = set(possible_grasp_verbs)
                                    
                                    # Check for overlap between predicted verb possibilities and ground truth
                                    matching_verbs = possible_grasp_verbs & gt_actions
                                    
                                    if matching_verbs:  # If there's any overlap
                                        # Take one matching verb (doesn't matter which one as they're equivalent)
                                        matched_verb = next(iter(matching_verbs))
                                        
                                        # Update action counts only once for this prediction
                                        self.pred_action_counts[matched_verb] += 1
                                        
                                        # Update pair counts
                                        pair_key = f"{grasp_instrument}_{matched_verb}"
                                        self.pred_instrument_action_pairs[pair_key] += 1
                                        
                                        # Store prediction
                                        prediction = {
                                            'instrument': {
                                                'name': grasp_instrument,
                                                'confidence': confidence
                                            },
                                            'action': {
                                                'name': matched_verb,
                                                'confidence': verb_confidence
                                            }
                                        }
                                        frame_predictions.append(prediction)
            
            # Update frame data
            frame_data = self.frame_data[frame_number]
            for pred in frame_predictions:
                inst_name = pred['instrument']['name']
                action_name = pred['action']['name']
                
                frame_data['pred_instruments'].append(inst_name)
                frame_data['pred_actions'].append(action_name)
                frame_data['pred_pairs'].append({
                    'instrument_name': inst_name,
                    'action': action_name
                })
            
            return frame_predictions
            
        except Exception as e:
            print(f"Error processing frame {frame_number}: {str(e)}")
            return []

    def calculate_metrics(self, category_data):
        """
        Calculate metrics using sklearn for a given category
        
        Args:
            category_data: Dict with 'gt' and 'pred' lists of labels
        Returns:
            Dict with precision, recall, f1, and ap scores
        """
        y_true = np.array(category_data['gt'])
        y_pred = np.array(category_data['pred'])
        
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        ap = average_precision_score(y_true, y_pred, average='macro')
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ap': ap
        }

    def prepare_binary_labels(self, category):
        """
        Prepare binary labels for a category (instruments, actions, or pairs)
        
        Args:
            category: String indicating which category to prepare
        Returns:
            Dict with 'gt' and 'pred' binary label arrays
        """
        if category == 'instruments':
            all_instruments = sorted(self.instrument_id_to_name.values())
            labels = {
                'gt': [],
                'pred': []
            }
            
            for frame_data in self.frame_data.values():
                gt_instruments = set(inst['name'] for inst in frame_data['gt_instruments'])
                pred_instruments = set(frame_data['pred_instruments'])
                
                # Create binary vectors for this frame
                for instrument in all_instruments:
                    labels['gt'].append(1 if instrument in gt_instruments else 0)
                    labels['pred'].append(1 if instrument in pred_instruments else 0)
                    
        elif category == 'actions':
            all_actions = sorted(self.action_id_to_name.values())
            labels = {
                'gt': [],
                'pred': []
            }
            
            for frame_data in self.frame_data.values():
                gt_actions = set()
                for inst in frame_data['gt_instruments']:
                    if 'valid_actions' in inst:
                        gt_actions.update(inst['valid_actions'])
                        
                pred_actions = set(frame_data['pred_actions'])
                
                # Create binary vectors for this frame
                for action in all_actions:
                    labels['gt'].append(1 if action in gt_actions else 0)
                    labels['pred'].append(1 if action in pred_actions else 0)
                    
        elif category == 'pairs':
            all_pairs = sorted(set(list(self.gt_instrument_action_pairs.keys()) + 
                                list(self.pred_instrument_action_pairs.keys())))
            labels = {
                'gt': [],
                'pred': []
            }
            
            for frame_data in self.frame_data.values():
                gt_pairs = set(f"{pair['instrument_name']}_{pair['action']}" 
                            for pair in frame_data['gt_pairs'])
                pred_pairs = set(f"{pair['instrument_name']}_{pair['action']}" 
                            for pair in frame_data['pred_pairs'])
                
                # Create binary vectors for this frame
                for pair in all_pairs:
                    labels['gt'].append(1 if pair in gt_pairs else 0)
                    labels['pred'].append(1 if pair in pred_pairs else 0)
                    
        return labels

    def print_statistics(self):
        """Print comprehensive statistics including precision, recall and AP"""
        print("\n" + "="*120)
        print("GROUND TRUTH AND PREDICTION STATISTICS")
        print("="*120)
        
        # Print instrument statistics with metrics
        print("\nINSTRUMENT INSTANCES:")
        print("-"*120)
        print(f"{'Instrument Type':<25} {'GT Count':<10} {'Pred Count':<10} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<10} {'Recall':<10}")
        print("-"*120)
        
        total_gt_instruments = 0
        total_pred_instruments = 0
        instrument_metrics = {}
        
        for instr in sorted(self.instrument_id_to_name.values()):
            gt_count = self.gt_instrument_counts[instr]
            pred_count = self.pred_instrument_counts[instr]
            
            # Calculate metrics
            tp = min(gt_count, pred_count)  # True Positives
            fp = max(0, pred_count - gt_count)  # False Positives
            fn = max(0, gt_count - pred_count)  # False Negatives
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            instrument_metrics[instr] = {
                'precision': precision,
                'recall': recall,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
            
            total_gt_instruments += gt_count
            total_pred_instruments += pred_count
            
            print(f"{instr:<25} {gt_count:<10} {pred_count:<10} {tp:<8} {fp:<8} {fn:<8} {precision:,.3f}    {recall:,.3f}")
        
        print("-"*120)
        # Calculate overall instrument metrics
        total_tp = sum(m['tp'] for m in instrument_metrics.values())
        total_fp = sum(m['fp'] for m in instrument_metrics.values())
        total_fn = sum(m['fn'] for m in instrument_metrics.values())
        total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        
        print(f"{'Total':<25} {total_gt_instruments:<10} {total_pred_instruments:<10} {total_tp:<8} {total_fp:<8} {total_fn:<8} {total_precision:,.3f}    {total_recall:,.3f}")
        
        # Print action statistics with metrics
        print("\nACTIONS:")
        print("-"*120)
        print(f"{'Action':<25} {'GT Count':<10} {'Pred Count':<10} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<10} {'Recall':<10}")
        print("-"*120)
        
        total_gt_actions = 0
        total_pred_actions = 0
        action_metrics = {}
        
        for action in sorted(self.action_id_to_name.values()):
            gt_count = self.gt_action_counts[action]
            pred_count = self.pred_action_counts[action]
            
            # Calculate metrics
            tp = min(gt_count, pred_count)
            fp = max(0, pred_count - gt_count)
            fn = max(0, gt_count - pred_count)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            action_metrics[action] = {
                'precision': precision,
                'recall': recall,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
            
            total_gt_actions += gt_count
            total_pred_actions += pred_count
            
            print(f"{action:<25} {gt_count:<10.1f} {pred_count:<10} {tp:<8.1f} {fp:<8.1f} {fn:<8.1f} {precision:,.3f}    {recall:,.3f}")
        
        print("-"*120)
        # Calculate overall action metrics
        total_tp = sum(m['tp'] for m in action_metrics.values())
        total_fp = sum(m['fp'] for m in action_metrics.values())
        total_fn = sum(m['fn'] for m in action_metrics.values())
        total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        
        print(f"{'Total':<25} {total_gt_actions:<10.1f} {total_pred_actions:<10} {total_tp:<8.1f} {total_fp:<8.1f} {total_fn:<8.1f} {total_precision:,.3f}    {total_recall:,.3f}")
        
        # Print pair statistics with metrics
        print("\nINSTRUMENT-ACTION PAIRS:")
        print("-"*120)
        print(f"{'Pair':<40} {'GT Count':<10} {'Pred Count':<10} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<10} {'Recall':<10}")
        print("-"*120)
        
        all_pairs = sorted(set(list(self.gt_instrument_action_pairs.keys()) + 
                            list(self.pred_instrument_action_pairs.keys())))
        
        total_gt_pairs = 0
        total_pred_pairs = 0
        pair_metrics = {}
        
        for pair in all_pairs:
            gt_count = self.gt_instrument_action_pairs[pair]
            pred_count = self.pred_instrument_action_pairs[pair]
            
            # Calculate metrics
            tp = min(gt_count, pred_count)
            fp = max(0, pred_count - gt_count)
            fn = max(0, gt_count - pred_count)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            pair_metrics[pair] = {
                'precision': precision,
                'recall': recall,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
            
            total_gt_pairs += gt_count
            total_pred_pairs += pred_count
            
            print(f"{pair:<40} {gt_count:<10.1f} {pred_count:<10} {tp:<8.1f} {fp:<8.1f} {fn:<8.1f} {precision:,.3f}    {recall:,.3f}")
        
        print("-"*120)
        # Calculate overall pair metrics
        total_tp = sum(m['tp'] for m in pair_metrics.values())
        total_fp = sum(m['fp'] for m in pair_metrics.values())
        total_fn = sum(m['fn'] for m in pair_metrics.values())
        total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        
        print(f"{'Total':<40} {total_gt_pairs:<10.1f} {total_pred_pairs:<10} {total_tp:<8.1f} {total_fp:<8.1f} {total_fn:<8.1f} {total_precision:,.3f}    {total_recall:,.3f}")
        
        # Print overall summary metrics
        print("\nOVERALL METRIC SUMMARY:")
        print("-"*50)
        categories = ["Instruments", "Actions", "Instrument-Action Pairs"]
        metrics = [
            (total_gt_instruments, total_pred_instruments, instrument_metrics),
            (total_gt_actions, total_pred_actions, action_metrics),
            (total_gt_pairs, total_pred_pairs, pair_metrics)
        ]
        
        for category, (gt_total, pred_total, category_metrics) in zip(categories, metrics):
            mean_precision = sum(m['precision'] for m in category_metrics.values()) / len(category_metrics) if category_metrics else 0
            mean_recall = sum(m['recall'] for m in category_metrics.values()) / len(category_metrics) if category_metrics else 0
            f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0
            
            print(f"\n{category}:")
            print(f"Mean Precision: {mean_precision:.3f}")
            print(f"Mean Recall: {mean_recall:.3f}")
            print(f"F1 Score: {f1_score:.3f}")

    def get_frame_annotations(self, frame_num):
        """Get annotations for a specific frame"""
        return self.frame_data.get(frame_num, {
            'gt_instruments': [],
            'gt_actions': [],
            'gt_pairs': [],
            'pred_instruments': [],
            'pred_actions': [],
            'pred_pairs': []
        })
def aggregate_metrics(all_metrics):
    """
    Aggregate metrics across multiple videos
    
    Args:
        all_metrics (dict): Dictionary of metrics for each video
    
    Returns:
        dict: Aggregated metrics across all videos
    """
    # Initialize aggregated metrics
    aggregated_metrics = {
        'instruments': {'gt': [], 'pred': []},
        'actions': {'gt': [], 'pred': []},
        'pairs': {'gt': [], 'pred': []}
    }
    
    # Combine metrics from all videos
    for video_metrics in all_metrics.values():
        # Aggregate instrument metrics
        aggregated_metrics['instruments']['gt'].extend(
            video_metrics['instrument_metrics']['gt']
        )
        aggregated_metrics['instruments']['pred'].extend(
            video_metrics['instrument_metrics']['pred']
        )
        
        # Aggregate action metrics
        aggregated_metrics['actions']['gt'].extend(
            video_metrics['action_metrics']['gt']
        )
        aggregated_metrics['actions']['pred'].extend(
            video_metrics['action_metrics']['pred']
        )
        
        # Aggregate pair metrics
        aggregated_metrics['pairs']['gt'].extend(
            video_metrics['pair_metrics']['gt']
        )
        aggregated_metrics['pairs']['pred'].extend(
            video_metrics['pair_metrics']['pred']
        )
    
    return aggregated_metrics

def calculate_combined_metrics(aggregated_metrics):
    """
    Calculate overall metrics from aggregated data
    
    Args:
        aggregated_metrics (dict): Aggregated metrics across all videos
    
    Returns:
        dict: Comprehensive metrics summary
    """
    metrics_summary = {}
    
    # Categories to analyze
    categories = ['instruments', 'actions', 'pairs']
    
    for category in categories:
        y_true = np.array(aggregated_metrics[category]['gt'])
        y_pred = np.array(aggregated_metrics[category]['pred'])
        
        # Calculate metrics
        metrics_summary[category] = {
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'average_precision': average_precision_score(y_true, y_pred, average='macro')
        }
    
    return metrics_summary

def main():
    try:
        print("\nStarting evaluation...")
        
        # Initialize ModelLoader
        loader = ModelLoader()
        print("\nInitializing models...")
        
        # Load models
        yolo_model = loader.load_yolo_model()
        verb_model = loader.load_verb_model()
        dataset_dir = loader.dataset_path
        
        # Initialize evaluator
        print("\nInitializing evaluator...")
        evaluator = GraSPEvaluator(
            yolo_model=yolo_model,
            verb_model=verb_model,
            dataset_dir=dataset_dir
        )
        
        # Process videos
        video_ids = ["VID41", "VID47", "VID50", "VID51", "VID53"]
        all_metrics = {}
        
        for video_id in video_ids:
            print(f"\nProcessing video: {video_id}")
            
            # Load ground truth first
            print("\nLoading ground truth annotations...")
            if not evaluator.load_ground_truth(video_id):
                raise Exception(f"Failed to load ground truth annotations for {video_id}")
            
            # Set up frames directory
            frames_dir = loader.dataset_path / "Videos" / video_id
            if not frames_dir.exists():
                raise Exception(f"Frames directory not found: {frames_dir}")
            
            # Get sorted frame files
            frame_files = sorted(frames_dir.glob("*.jpg"), 
                               key=lambda x: int(x.stem))
            
            if not frame_files:
                raise Exception(f"No frames found in directory: {frames_dir}")
            
            print(f"\nFound {len(frame_files)} frames to process")
            print("\nProcessing frames...")
            
            # Process each frame
            with tqdm(total=len(frame_files)) as pbar:
                for frame_file in frame_files:
                    frame_num = int(frame_file.stem)
                    
                    try:
                        # Get annotations for current frame
                        frame_annotations = evaluator.get_frame_annotations(frame_num)
                        
                        # Process frame and get predictions
                        evaluator.evaluate_frame(
                            img_path=str(frame_file),
                            frame_annotations=frame_annotations,
                            save_visualization=False
                        )
                        
                    except Exception as e:
                        print(f"\nWarning: Error processing frame {frame_num}: {str(e)}")
                        continue
                        
                    pbar.update(1)
            
            # Print statistics for each video
            print("\nGenerating statistics for video:", video_id)
            evaluator.print_statistics()
            
            # Store metrics for this video
            all_metrics[video_id] = {
                'instrument_metrics': evaluator.prepare_binary_labels('instruments'),
                'action_metrics': evaluator.prepare_binary_labels('actions'),
                'pair_metrics': evaluator.prepare_binary_labels('pairs')
            }
        
        # Combine metrics across all videos
        print("\nCalculating combined metrics across all videos...")
        aggregated_metrics = aggregate_metrics(all_metrics)
        combined_metrics = calculate_combined_metrics(aggregated_metrics)
        
        # Print combined metrics
        print("\nCOMBINED METRICS ACROSS ALL VIDEOS:")
        print("-"*50)
        for category, metrics in combined_metrics.items():
            print(f"\n{category.upper()} METRICS:")
            for metric, value in metrics.items():
                print(f"{metric.capitalize()}: {value:.4f}")
        
        print("\nEvaluation complete!")
        return True
        
    except Exception as e:
        print(f"\nError in evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\nStarting GraSP evaluation...")
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")