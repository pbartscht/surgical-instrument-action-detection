

"""
Evaluation Script for CholecT50 Multitask Model on HeiChole Dataset
Unified version that loads and evaluates ground truth consistently
"""

import os
import torch
import pytorch_lightning as pl
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from mulit_model.cholect50_multitask_model import CholecT50Model

# Global mappings
TOOL_MAPPING = {
    0: 'grasper', 1: 'bipolar', 2: 'hook', 
    3: 'scissors', 4: 'clipper', 5: 'irrigator'
}

IGNORED_INSTRUMENTS = {
    6: 'specimen_bag'  # Index: Name der zu ignorierenden Instrumente
}

VERB_MAPPING = {
    0: 'grasp', 1: 'retract', 2: 'dissect', 3: 'coagulate', 
    4: 'clip', 5: 'cut', 6: 'aspirate', 7: 'irrigate', 
    8: 'pack', 9: 'null_verb'
}


# Constants for Dataset Mappings
CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING = {
    'grasper': 'grasper',
    'bipolar': 'coagulation',
    'clipper': 'clipper',
    'hook': 'coagulation',  # revised: hook is also a coagulation instrument
    'scissors': 'scissors',
    'irrigator': 'suction_irrigation'
}

CHOLECT50_TO_HEICHOLE_VERB_MAPPING = {
    'grasp': 'grasp',
    'retract': 'hold',
    'dissect': 'hold',
    'coagulate': 'hold',
    'clip': 'clip',
    'cut': 'cut',
    'irrigate': 'suction_irrigation',
    'aspirate': 'suction_irrigation',
    'pack': 'hold',
    'null_verb': 'hold'
}

HEICHOLE_SPECIFIC_INSTRUMENTS = {
    'specimen_bag',
    'stapler'
}

class HeiCholeEvaluator:
    def __init__(self, model_path, dataset_path, confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CholecT50Model.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.dataset_path = Path(dataset_path)
        self.confidence_threshold = confidence_threshold
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define fixed order of labels matching ground truth JSON structure
        self.instrument_labels = [
            'grasper', 'clipper', 'coagulation', 'scissors',
            'suction_irrigation', 'specimen_bag', 'stapler'
        ]
        
        self.action_labels = [
            'grasp', 'hold', 'cut', 'clip'
        ]

    def load_ground_truth(self, video):
        """
        Loads ground truth annotations for a specific video.
        Consistent with the YOLO+Verb evaluation method.
        """
        frame_annotations = defaultdict(lambda: {
            'instruments': defaultdict(int),
            'actions': defaultdict(int)
        })
        
        labels_folder = self.dataset_path / "Labels"
        json_file = labels_folder / f"{video}.json"
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                frames = data.get('frames', {})
                
                for frame_num, frame_data in frames.items():
                    frame_number = int(frame_num)
                    
                    # Process instruments (binary)
                    instruments = frame_data.get('instruments', {})
                    for instr_name, present in instruments.items():
                        frame_annotations[frame_number]['instruments'][instr_name] = 1 if present > 0 else 0
                    
                    # Process actions (binary)
                    actions = frame_data.get('actions', {})
                    for action_name, present in actions.items():
                        frame_annotations[frame_number]['actions'][action_name] = 1 if present > 0 else 0
                
                return frame_annotations
                
        except Exception as e:
            print(f"Error loading annotations: {str(e)}")
            raise

    def analyze_label_distribution(self, videos):
        """
        Analyzes the distribution of ground truth labels across all videos
        """
        frequencies = {
            'instruments': defaultdict(int),
            'actions': defaultdict(int)
        }
        
        total_frames = 0
        
        print("\nAnalyzing ground truth label distribution...")
        
        for video in videos:
            json_file = self.dataset_path / "Labels" / f"{video}.json"
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    frames = data.get('frames', {})
                    total_frames += len(frames)
                    
                    # Count occurrences
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
        
        return frequencies, total_frames

    def evaluate_frame(self, image_path):
        """Evaluate a single frame with the multitask model"""
        img = Image.open(image_path)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            iv_output, tool_output, verb_output, _ = self.model(img_tensor)
            
            # Apply sigmoid to get probabilities
            tool_probs = torch.sigmoid(tool_output)[0]
            verb_probs = torch.sigmoid(verb_output)[0]
            iv_probs = torch.sigmoid(iv_output)[0]
            
            # Convert predictions to HeiChole format
            predictions = {
                'instruments': defaultdict(float),
                'actions': defaultdict(float)
            }
            
            # Process tool predictions
            for idx, prob in enumerate(tool_probs):
                if idx in TOOL_MAPPING:
                    cholect50_tool = TOOL_MAPPING[idx]
                    if cholect50_tool in CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING:
                        heichole_tool = CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING[cholect50_tool]
                        predictions['instruments'][heichole_tool] = max(
                            predictions['instruments'][heichole_tool],
                            float(prob)
                        )
            
            # Process verb predictions
            for idx, prob in enumerate(verb_probs):
                if idx in VERB_MAPPING:
                    cholect50_verb = VERB_MAPPING[idx]
                    if cholect50_verb in CHOLECT50_TO_HEICHOLE_VERB_MAPPING:
                        heichole_verb = CHOLECT50_TO_HEICHOLE_VERB_MAPPING[cholect50_verb]
                        predictions['actions'][heichole_verb] = max(
                            predictions['actions'][heichole_verb],
                            float(prob)
                        )
            
            return predictions

    def calculate_metrics(self, predictions_per_frame, ground_truth):
        """
        Calculate metrics per video, consistent with YOLO+Verb evaluation
        """
        results = {
            'instruments': {'per_class': {}, 'mean_metrics': {}},
            'actions': {'per_class': {}, 'mean_metrics': {}}
        }
        
        all_frame_numbers = sorted(list(ground_truth.keys()))
        num_frames = len(all_frame_numbers)
        frame_to_idx = {frame: idx for idx, frame in enumerate(all_frame_numbers)}
        
        for category in ['instruments', 'actions']:
            label_list = self.instrument_labels if category == 'instruments' else self.action_labels
            
            # Initialize matrices
            y_true = np.zeros((num_frames, len(label_list)))
            y_pred = np.zeros((num_frames, len(label_list)))
            y_scores = np.zeros((num_frames, len(label_list)))
            
            # Fill ground truth matrix
            for frame_num, frame_data in ground_truth.items():
                frame_idx = frame_to_idx[frame_num]
                for label_idx, label in enumerate(label_list):
                    if label in frame_data[category]:
                        y_true[frame_idx, label_idx] = frame_data[category][label]
            
            # Fill prediction matrix
            for frame_num, preds in predictions_per_frame.items():
                if frame_num in frame_to_idx:
                    frame_idx = frame_to_idx[frame_num]
                    for label_idx, label in enumerate(label_list):
                        confidence = preds[category].get(label, 0.0)
                        y_scores[frame_idx, label_idx] = confidence
                        y_pred[frame_idx, label_idx] = 1 if confidence >= self.confidence_threshold else 0
            
            # Calculate per-class metrics
            for i, label in enumerate(label_list):
                support = int(np.sum(y_true[:, i]))
                if support > 0:  # Only calculate metrics if class exists in ground truth
                    ap = average_precision_score(y_true[:, i], y_scores[:, i])
                    f1 = f1_score(y_true[:, i], y_pred[:, i])
                    precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
                    recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
                    
                    results[category]['per_class'][label] = {
                        'ap': ap,
                        'f1': f1,
                        'precision': precision,
                        'recall': recall,
                        'support': support,
                        'predictions': int(np.sum(y_pred[:, i]))
                    }
            
            # Calculate mean metrics only for classes with support
            valid_classes = [m for m in results[category]['per_class'].values() if m['support'] > 0]
            if valid_classes:
                results[category]['mean_metrics'] = {
                    'mAP': np.mean([m['ap'] for m in valid_classes]),
                    'mean_f1': np.mean([m['f1'] for m in valid_classes]),
                    'mean_precision': np.mean([m['precision'] for m in valid_classes]),
                    'mean_recall': np.mean([m['recall'] for m in valid_classes])
                }
        
        return results

def main():
    # Get base directory
    base_dir = Path(__file__).resolve().parent.parent.parent.parent  # Navigate up to project root
    
    # Configuration with resolved paths
    model_path = base_dir / "models" / "multitask-surgical-workflow" / "checkpoints" / "clean-dust-36" / "clean-dust-36-epoch-epoch=26.ckpt"
    dataset_path = Path("/data/Bartscht/HeiChole/domain_adaptation/test")
    
    print(f"\nEvaluating model: {model_path}")
    print(f"Dataset path: {dataset_path}")
    
    # Verify that model path exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        
    # Initialize evaluator
    evaluator = HeiCholeEvaluator(model_path, dataset_path)
    
    # Get video list
    labels_dir = dataset_path / "Labels"
    videos = [f.stem for f in labels_dir.glob("*.json")]
    print(f"\nFound {len(videos)} videos to analyze: {', '.join(videos)}")
    
    # Analyze ground truth distribution
    frequencies, total_frames = evaluator.analyze_label_distribution(videos)
    
    # Process each video
    all_predictions = {}
    all_ground_truth = {}
    
    for video in videos:
        print(f"\nProcessing video: {video}")
        
        # Load ground truth for this video
        video_ground_truth = evaluator.load_ground_truth(video)
        for frame_num, gt in video_ground_truth.items():
            # Erzeuge einen eindeutigen Schlüssel, z.B. "VID13_23"
            key = f"{video}_{frame_num}"
            all_ground_truth[key] = gt
        
        # Process each frame
        video_dir = dataset_path / "Videos" / video
        for image_path in video_dir.glob("*.png"):
            frame_num = int(image_path.stem)
            predictions = evaluator.evaluate_frame(image_path)
            key = f"{video}_{frame_num}"
            all_predictions[key] = predictions
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(all_predictions, all_ground_truth)
    
    # Print detailed results
    print("\n====== GROUND TRUTH DISTRIBUTION ======")
    for category in ['instruments', 'actions']:
        print(f"\n{category.upper()}:")
        print(f"{'Label':20s} {'Count':>10s} {'% of Frames':>12s}")
        print("-" * 50)
        
        for label, count in sorted(frequencies[category].items()):
            percentage = (count / total_frames) * 100
            print(f"{label:20s} {count:10d} {percentage:11.2f}%")
    
    print("\n====== MODEL PERFORMANCE ======")
    for category in ['instruments', 'actions']:
        print(f"\n{category.upper()}:")
        print(f"{'Label':20s} {'AP':>10s} {'F1':>10s} {'Prec':>10s} {'Rec':>10s} {'Support':>10s}")
        print("-" * 80)
        
        for label, scores in metrics[category]['per_class'].items():
            print(f"{label:20s} {scores['ap']:10.4f} {scores['f1']:10.4f} "
                  f"{scores['precision']:10.4f} {scores['recall']:10.4f} {scores['support']:10d}")
        
        print(f"\nMean Metrics for {category}:")
        means = metrics[category]['mean_metrics']
        print(f"mAP: {means['mAP']:.4f}")
        print(f"F1:  {means['mean_f1']:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise
