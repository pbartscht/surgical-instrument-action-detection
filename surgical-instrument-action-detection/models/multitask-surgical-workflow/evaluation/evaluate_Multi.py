import os
from tqdm.auto import tqdm
import torch
import pytorch_lightning as pl
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score
from mulit_model.cholect50_multitask_model import CholecT50Model

# Constants and Mappings
TOOL_MAPPING = {
    0: 'grasper', 1: 'bipolar', 2: 'hook', 
    3: 'scissors', 4: 'clipper', 5: 'irrigator'
}

VERB_MAPPING = {
    0: 'grasp', 1: 'retract', 2: 'dissect', 3: 'coagulate', 
    4: 'clip', 5: 'cut', 6: 'aspirate', 7: 'irrigate', 
    8: 'pack', 9: 'null_verb'
}

class IVPairMapper:
    def __init__(self):
        self.iv_pairs = {
            0: ('grasper', 'dissect'),
            1: ('grasper', 'grasp'),
            2: ('grasper', 'pack'),
            3: ('grasper', 'retract'),
            4: ('bipolar', 'coagulate'),
            5: ('bipolar', 'dissect'),
            6: ('bipolar', 'grasp'),
            7: ('bipolar', 'retract'),
            8: ('hook', 'coagulate'),
            9: ('hook', 'cut'),
            10: ('hook', 'dissect'),
            11: ('hook', 'retract'),
            12: ('scissors', 'coagulate'),
            13: ('scissors', 'cut'),
            14: ('scissors', 'dissect'),
            15: ('clipper', 'clip'),
            16: ('irrigator', 'aspirate'),
            17: ('irrigator', 'dissect'),
            18: ('irrigator', 'irrigate'),
            19: ('irrigator', 'retract'),
            20: ('grasper', 'null_verb'),
            21: ('bipolar', 'null_verb'),
            22: ('hook', 'null_verb'),
            23: ('scissors', 'null_verb'),
            24: ('clipper', 'null_verb'),
            25: ('irrigator', 'null_verb')
        }
        self.pair_to_index = {pair: idx for idx, pair in self.iv_pairs.items()}
    
    def get_iv_pair(self, index):
        return self.iv_pairs.get(index)
    
    def get_iv_index(self, instrument, verb):
        return self.pair_to_index.get((instrument, verb))

class CholecT50Evaluator:
    def __init__(self, model_path, dataset_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CholecT50Model.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.dataset_dir = Path(dataset_dir)
        self.iv_mapper = IVPairMapper()
        
        # Initialize prediction and ground truth storage
        self.predictions = {
            'tools': defaultdict(list),
            'verbs': defaultdict(list),
            'iv_pairs': defaultdict(list)
        }
        self.ground_truth = {
            'tools': defaultdict(list),
            'verbs': defaultdict(list),
            'iv_pairs': defaultdict(list)
        }
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_ground_truth(self, video_id):
        """Load ground truth annotations for a video"""
        json_file = self.dataset_dir / "labels" / f"{video_id}.json"
        frame_annotations = defaultdict(lambda: {
            'tools': np.zeros(len(TOOL_MAPPING)),
            'verbs': np.zeros(len(VERB_MAPPING)),
            'iv_pairs': np.zeros(len(self.iv_mapper.iv_pairs))
        })
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            annotations = data.get('annotations', {})
            
            for frame, instances in annotations.items():
                frame_num = int(frame)
                
                for instance in instances:
                    tool_idx = instance[1]
                    verb_idx = instance[7]
                    
                    if 0 <= tool_idx < len(TOOL_MAPPING) and 0 <= verb_idx < len(VERB_MAPPING):
                        # Set tool and verb ground truth
                        frame_annotations[frame_num]['tools'][tool_idx] = 1
                        frame_annotations[frame_num]['verbs'][verb_idx] = 1
                        
                        # Set IV pair ground truth
                        tool_name = TOOL_MAPPING[tool_idx]
                        verb_name = VERB_MAPPING[verb_idx]
                        iv_idx = self.iv_mapper.get_iv_index(tool_name, verb_name)
                        if iv_idx is not None:
                            frame_annotations[frame_num]['iv_pairs'][iv_idx] = 1
        
        return frame_annotations

    def evaluate_frame(self, img_path, ground_truth):
        """Evaluate a single frame"""
        try:
            # Load and preprocess image
            img = Image.open(img_path)
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                iv_output, tool_output, verb_output, _ = self.model(img_tensor)
                
                # Get probabilities
                tool_probs = torch.sigmoid(tool_output)[0].cpu().numpy()
                verb_probs = torch.sigmoid(verb_output)[0].cpu().numpy()
                iv_probs = torch.sigmoid(iv_output)[0].cpu().numpy()
                
                # Store predictions and ground truth for tools
                for i in range(len(TOOL_MAPPING)):
                    tool_name = TOOL_MAPPING[i]
                    self.predictions['tools'][tool_name].append(tool_probs[i])
                    self.ground_truth['tools'][tool_name].append(ground_truth['tools'][i])
                
                # Store predictions and ground truth for verbs
                for i in range(len(VERB_MAPPING)):
                    verb_name = VERB_MAPPING[i]
                    self.predictions['verbs'][verb_name].append(verb_probs[i])
                    self.ground_truth['verbs'][verb_name].append(ground_truth['verbs'][i])
                
                # Store predictions and ground truth for IV pairs
                for i in range(len(self.iv_mapper.iv_pairs)):
                    tool, verb = self.iv_mapper.get_iv_pair(i)
                    pair_name = f"{tool}_{verb}"
                    self.predictions['iv_pairs'][pair_name].append(iv_probs[i])
                    self.ground_truth['iv_pairs'][pair_name].append(ground_truth['iv_pairs'][i])
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")

    
    def calculate_metrics(self):
        """Calculate all metrics for all categories"""
        results = {}
        
        for category in ['tools', 'verbs', 'iv_pairs']:
            category_metrics = {}
            total_support = 0
            
            # Calculate total support for distribution
            for item in self.predictions[category]:
                y_true = np.array(self.ground_truth[category][item])
                total_support += np.sum(y_true)
            
            for item in self.predictions[category]:
                y_true = np.array(self.ground_truth[category][item])
                y_score = np.array(self.predictions[category][item])
                
                # Count predictions with threshold 0.5
                y_pred = (y_score >= 0.5).astype(int)
                n_pred = np.sum(y_pred)
                support = int(np.sum(y_true))
                
                if support > 0:
                    ap = average_precision_score(y_true, y_score)
                    precision = np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_pred == 1) + 1e-10)
                    recall = np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-10)
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
                else:
                    ap = precision = recall = f1 = 0.0
                
                category_metrics[item] = {
                    'AP': ap,
                    'support': support,
                    'dist': (support / total_support * 100) if total_support > 0 else 0,
                    'pred': n_pred,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
            
            # Calculate means
            valid_metrics = [m for m in category_metrics.values() if m['support'] > 0]
            results[category] = {
                'per_class': category_metrics,
                'mean': {
                    'AP': np.mean([m['AP'] for m in valid_metrics]),
                    'f1': np.mean([m['f1'] for m in valid_metrics])
                },
                'weighted_mean': {
                    'AP': np.average([m['AP'] for m in valid_metrics], weights=[m['support'] for m in valid_metrics]),
                    'f1': np.average([m['f1'] for m in valid_metrics], weights=[m['support'] for m in valid_metrics])
                }
            }
        
        return results

    def print_results(self, results):
        """Print all metrics in a simple format"""
        print("\nEvaluation Results:")
        print("="*80)
        
        for category in ['tools', 'verbs', 'iv_pairs']:
            print(f"\n{category.upper()}:")
            print("-"*80)
            print(f"{'Item':<20} {'Support':>8} {'Dist%':>8} {'Pred':>8} {'AP':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
            print("-"*80)
            
            for item, metrics in results[category]['per_class'].items():
                print(f"{item:<20} {metrics['support']:>8d} {metrics['dist']:>8.2f} {metrics['pred']:>8d} "
                    f"{metrics['AP']:>8.4f} {metrics['f1']:>8.4f} {metrics['precision']:>8.4f} {metrics['recall']:>8.4f}")
            
            print("-"*80)
            print(f"Mean         {'-':>8} {'-':>8} {'-':>8} "
                f"{results[category]['mean']['AP']:>8.4f} {results[category]['mean']['f1']:>8.4f}")
            print(f"Weighted Mean{'-':>8} {'-':>8} {'-':>8} "
                f"{results[category]['weighted_mean']['AP']:>8.4f} {results[category]['weighted_mean']['f1']:>8.4f}")
def main():
    try:
        print("\nStarting CholecT50 evaluation...")
        
        # Configure paths
        base_dir = Path(__file__).resolve().parent
        model_path = Path( "/home/Bartscht/YOLO/surgical-instrument-action-detection/models/multitask-surgical-workflow/checkpoints/clean-dust-36/last.ckpt")
        dataset_dir = Path("/data/Bartscht/CholecT50")
        
        # Initialize evaluator
        evaluator = CholecT50Evaluator(model_path, dataset_dir)
        
        # Process videos (example video IDs from CholecT50)
        video_ids = ["VID92", "VID96", "VID103", "VID110", "VID111"]
        
        for video_id in video_ids:
            print(f"\nProcessing {video_id}...")
            
            # Load ground truth
            ground_truth = evaluator.load_ground_truth(video_id)
            
            # Process frames
            frames_dir = evaluator.dataset_dir / "videos" / video_id
            frame_files = sorted(frames_dir.glob("*.png"))
            
            for frame_file in tqdm(frame_files, desc=f"Evaluating {video_id}"):
                frame_num = int(frame_file.stem)
                if frame_num in ground_truth:
                    evaluator.evaluate_frame(
                        str(frame_file),
                        ground_truth[frame_num]
                    )
        
        # Calculate and print final results
        results = evaluator.calculate_metrics()
        evaluator.print_results(results)
        
        print("\nEvaluation complete!")
        return results
        
    except Exception as e:
        print(f"\nError in evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()