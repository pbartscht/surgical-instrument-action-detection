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

    def print_statistics(self):
        """Print comprehensive statistics comparing ground truth and predictions"""
        print("\n" + "="*100)
        print("GROUND TRUTH AND PREDICTION STATISTICS")
        print("="*100)
        
        # Print instrument statistics
        print("\nINSTRUMENT INSTANCES:")
        print("-"*80)
        print(f"{'Instrument Type':<30} {'GT Count':<15} {'Pred Count':<15} {'Difference':<15}")
        print("-"*80)
        
        total_gt_instruments = 0
        total_pred_instruments = 0
        
        for instr in sorted(self.instrument_id_to_name.values()):
            gt_count = self.gt_instrument_counts[instr]
            pred_count = self.pred_instrument_counts[instr]
            diff = pred_count - gt_count
            total_gt_instruments += gt_count
            total_pred_instruments += pred_count
            print(f"{instr:<30} {gt_count:<15} {pred_count:<15} {diff:+<15}")
            
        print("-"*80)
        total_diff = total_pred_instruments - total_gt_instruments
        print(f"{'Total':<30} {total_gt_instruments:<15} {total_pred_instruments:<15} {total_diff:+<15}")
        
        # Print action statistics
        print("\nACTIONS:")
        print("-"*80)
        print(f"{'Action':<30} {'GT Count':<15} {'Pred Count':<15} {'Difference':<15}")
        print("-"*80)
        
        total_gt_actions = 0
        total_pred_actions = 0
        
        for action in sorted(self.action_id_to_name.values()):
            gt_count = self.gt_action_counts[action]
            pred_count = self.pred_action_counts[action]
            diff = pred_count - gt_count
            total_gt_actions += gt_count
            total_pred_actions += pred_count
            print(f"{action:<30} {gt_count:<15.1f} {pred_count:<15} {diff:+.1f}")
            
        print("-"*80)
        total_diff = total_pred_actions - total_gt_actions
        print(f"{'Total':<30} {total_gt_actions:<15.1f} {total_pred_actions:<15} {total_diff:+.1f}")
        
        # Print pair statistics
        print("\nINSTRUMENT-ACTION PAIRS:")
        print("-"*100)
        print(f"{'Pair':<50} {'GT Count':<15} {'Pred Count':<15} {'Difference':<15}")
        print("-"*100)
        
        # Combine all pair keys
        all_pairs = sorted(set(list(self.gt_instrument_action_pairs.keys()) + 
                              list(self.pred_instrument_action_pairs.keys())))
        
        total_gt_pairs = 0
        total_pred_pairs = 0
        
        for pair in all_pairs:
            gt_count = self.gt_instrument_action_pairs[pair]
            pred_count = self.pred_instrument_action_pairs[pair]
            diff = pred_count - gt_count
            total_gt_pairs += gt_count
            total_pred_pairs += pred_count
            print(f"{pair:<50} {gt_count:<15.1f} {pred_count:<15} {diff:+.1f}")
            
        print("-"*100)
        total_diff = total_pred_pairs - total_gt_pairs
        print(f"{'Total':<50} {total_gt_pairs:<15.1f} {total_pred_pairs:<15} {total_diff:+.1f}")
        
        # Print frame-wise summary
        print("\nFRAME-WISE SUMMARY:")
        print(f"Total frames processed: {len(self.frame_data)}")
        print(f"Average GT instruments per frame: {total_gt_instruments/len(self.frame_data):.2f}")
        print(f"Average predicted instruments per frame: {total_pred_instruments/len(self.frame_data):.2f}")
        print(f"Average GT actions per frame: {total_gt_actions/len(self.frame_data):.2f}")
        print(f"Average predicted actions per frame: {total_pred_actions/len(self.frame_data):.2f}")

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
        
        # Process video
        video_id = "VID41"
        print(f"\nProcessing video: {video_id}")
        
        # Load ground truth first
        print("\nLoading ground truth annotations...")
        if not evaluator.load_ground_truth(video_id):
            raise Exception("Failed to load ground truth annotations")
        
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
                
                # Get annotations for current frame
                frame_annotations = evaluator.get_frame_annotations(frame_num)
                
                # Process frame and get predictions
                predictions = evaluator.evaluate_frame(
                    img_path=str(frame_file),
                    frame_annotations=frame_annotations,
                    save_visualization=False  # Set to True if you want visualizations
                )
                
                pbar.update(1)
        
        # Print final statistics
        print("\nGenerating final statistics...")
        evaluator.print_statistics()
        
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