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
CONFIDENCE_THRESHOLD = 0.1
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
    'coagulate': ['Cauterize'],
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
        

class GTLoader:
    def __init__(self):
        # Mapping for instrument IDs to names
        self.instrument_id_to_name = {
            1: 'Bipolar Forceps',
            2: 'Prograsp Forceps',
            3: 'Large Needle Driver',
            4: 'Monopolar Curved Scissors',
            5: 'Suction Instrument',
            6: 'Clip Applier',
            7: 'Laparoscopic Grasper'
        }
        
        # Mapping for action IDs to names
        self.action_id_to_name = {
            1: 'Cauterize',
            2: 'Close',
            3: 'Cut',
            4: 'Grasp',
            5: 'Hold',
            6: 'Open',
            7: 'Open Something',
            8: 'Pull',
            9: 'Push',
            10: 'Release',
            11: 'Still',
            12: 'Suction',
            13: 'Travel',
            14: 'Other'
        }
        
        # Initialize counters and data structures
        self.reset_counters()

    def reset_counters(self):
        """Reset all counters and data structures"""
        self.instrument_counts = {name: 0 for name in self.instrument_id_to_name.values()}
        self.action_counts = {name: 0 for name in self.action_id_to_name.values()}
        self.instrument_action_pairs = defaultdict(int)
        
        # Store detailed frame-wise data
        self.frame_data = defaultdict(lambda: {
            'instruments': [],  # List of dicts with instrument info
            'actions': [],     # List of dicts with action info
            'pairs': []        # List of instrument-action pairs
        })

    def load_video_annotations(self, json_path):
        """Load and process ground truth annotations from a video's JSON file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            print(f"\nLoading annotations from: {json_path}")
            self.reset_counters()
            
            # Process each frame
            for frame_id, frame_data in data['frames'].items():
                frame_num = int(frame_id.split('.')[0])  # Extract frame number
                
                # Process each instrument instance in the frame
                for instrument in frame_data.get('instruments', []):
                    instance_id = instrument.get('id')
                    category_id = instrument.get('category_id')
                    
                    if category_id in self.instrument_id_to_name:
                        instrument_name = self.instrument_id_to_name[category_id]
                        
                        # Count this instrument instance
                        self.instrument_counts[instrument_name] += 1
                        
                        # Store instrument instance info
                        instrument_info = {
                            'instance_id': instance_id,
                            'name': instrument_name,
                            'bbox': instrument.get('bbox', []),
                            'category_id': category_id
                        }
                        
                        # Get valid actions for this instance
                        actions = instrument.get('actions', [])
                        if isinstance(actions, list):
                            valid_actions = []
                            for action_id in actions:
                                if action_id in self.action_id_to_name:
                                    action_name = self.action_id_to_name[action_id]
                                    valid_actions.append(action_name)
                                    
                                    # Count each action (normalized by number of valid actions)
                                    self.action_counts[action_name] += 1/len(actions)
                                    
                                    # Count instrument-action pair
                                    pair_key = f"{instrument_name}_{action_name}"
                                    self.instrument_action_pairs[pair_key] += 1/len(actions)
                            
                            # Store action info for this instance
                            if valid_actions:
                                instrument_info['valid_actions'] = valid_actions
                        
                        # Add to frame data
                        self.frame_data[frame_num]['instruments'].append(instrument_info)
                        
                        # Add instrument-action pairs for this frame
                        for action in valid_actions:
                            self.frame_data[frame_num]['pairs'].append({
                                'instrument_id': instance_id,
                                'instrument_name': instrument_name,
                                'action': action
                            })
            
            return True
            
        except Exception as e:
            print(f"Error loading annotations: {str(e)}")
            return False

    def print_statistics(self):
        """Print comprehensive ground truth statistics"""
        print("\n" + "="*80)
        print("GROUND TRUTH STATISTICS")
        print("="*80)
        
        print("\nINSTRUMENT INSTANCES:")
        print("-"*60)
        print(f"{'Instrument Type':<30} {'Count':<10}")
        print("-"*60)
        
        total_instruments = 0
        for instr, count in sorted(self.instrument_counts.items()):
            print(f"{instr:<30} {count:<10}")
            total_instruments += count
        print("-"*60)
        print(f"{'Total Instruments':<30} {total_instruments:<10}")
        
        print("\nACTIONS (Normalized for multiple valid actions per instance):")
        print("-"*60)
        print(f"{'Action':<25} {'Count':<10}")
        print("-"*60)
        
        total_actions = 0
        for action, count in sorted(self.action_counts.items()):
            print(f"{action:<25} {count:<10.1f}")
            total_actions += count
        print("-"*60)
        print(f"{'Total Actions':<25} {total_actions:<10.1f}")
        
        print("\nINSTRUMENT-ACTION PAIRS (Normalized):")
        print("-"*80)
        print(f"{'Instrument-Action Pair':<50} {'Count':<10}")
        print("-"*80)
        
        total_pairs = 0
        for pair, count in sorted(self.instrument_action_pairs.items()):
            print(f"{pair:<50} {count:<10.1f}")
            total_pairs += count
        print("-"*80)
        print(f"{'Total Pairs':<50} {total_pairs:<10.1f}")

    def get_frame_annotations(self, frame_num):
        """Get all annotations for a specific frame"""
        return self.frame_data.get(frame_num, {
            'instruments': [],
            'actions': [],
            'pairs': []
        })
    

def main():
    try:
        print("\nStarting evaluation...")
        
        # Initialize ModelLoader
        loader = GTLoader()
        print("\nInitializing models...")
        
        # Load models
        #yolo_model = loader.load_yolo_model()
        #verb_model = loader.load_verb_model()
        #dataset_dir = loader.dataset_path
        
        loader.load_video_annotations("/data/Bartscht/GrasP/test/Labels/VID41.json")
        loader.print_statistics()

        #evaluator = GraSPEvaluator(yolo_model, verb_model, dataset_dir)
        #evaluator.load_video_annotations("VID41")
        # Nach der Verarbeitung aller Frames mit evaluate_frame
        #evaluator.print_statistics()
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()