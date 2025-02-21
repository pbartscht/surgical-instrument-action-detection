

"""
Evaluation Script for CholecT50 Multitask Model on GraSP Dataset
Unified version that loads and evaluates ground truth consistently
"""

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
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from mulit_model.cholect50_multitask_model import CholecT50Model

# Constants and Mappings
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

CHOLECT50_TO_GRASP_VERB_MAPPING = {
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

class GraSPMultitaskEvaluator:
    def __init__(self, model_path, dataset_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CholecT50Model.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.dataset_dir = Path(dataset_dir)
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # GraSP specific mappings
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

    def generate_video_report(self, video_id):
        """Generate a detailed report for a specific video"""
        print(f"\n{'='*40} Video Report: {video_id} {'='*40}")
        
        # Print frame-wise statistics
        print("\nFrame-wise Statistics:")
        total_frames = len(self.frame_data)
        frames_with_instruments = sum(1 for frame in self.frame_data.values() if frame['gt_instruments'])
        frames_with_actions = sum(1 for frame in self.frame_data.values() if frame['gt_actions'])
        
        print(f"Total Frames: {total_frames}")
        print(f"Frames with Instruments: {frames_with_instruments} ({frames_with_instruments/total_frames*100:.1f}%)")
        print(f"Frames with Actions: {frames_with_actions} ({frames_with_actions/total_frames*100:.1f}%)")
        
        # Calculate and return metrics
        return {
            'frame_stats': {
                'total_frames': total_frames,
                'frames_with_instruments': frames_with_instruments,
                'frames_with_actions': frames_with_actions
            }
        }

    def aggregate_metrics_across_videos(self, all_metrics):
        """
        Aggregate metrics across multiple videos
        
        Args:
            all_metrics (dict): Dictionary of metrics for each video
        
        Returns:
            dict: Aggregated metrics across all videos
        """
        aggregated = {
            'instruments': {'gt': [], 'pred': []},
            'actions': {'gt': [], 'pred': []},
            'pairs': {'gt': [], 'pred': []}
        }
        
        # Dictionary to map category names to metric keys
        metric_keys = {
            'instruments': 'instrument_metrics',
            'actions': 'action_metrics',
            'pairs': 'pair_metrics'
        }
        
        # Combine metrics from all videos
        for video_metrics in all_metrics.values():
            for category in ['instruments', 'actions', 'pairs']:
                metric_key = metric_keys[category]
                aggregated[category]['gt'].extend(video_metrics[metric_key]['gt'])
                aggregated[category]['pred'].extend(video_metrics[metric_key]['pred'])
        
        # Calculate final metrics
        final_metrics = {}
        for category in aggregated:
            y_true = np.array(aggregated[category]['gt'])
            y_pred = np.array(aggregated[category]['pred'])
            
            final_metrics[category] = {
                'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'ap': average_precision_score(y_true, y_pred, average='macro')
            }
        
        return final_metrics

    def reset_counters(self):
        """Reset all counters and data structures"""
        self.gt_instrument_counts = {name: 0 for name in self.instrument_id_to_name.values()}
        self.gt_action_counts = {name: 0 for name in self.action_id_to_name.values()}
        self.gt_instrument_action_pairs = defaultdict(int)
        
        self.pred_instrument_counts = {name: 0 for name in self.instrument_id_to_name.values()}
        self.pred_action_counts = {name: 0 for name in self.action_id_to_name.values()}
        self.pred_instrument_action_pairs = defaultdict(int)
        
        self.frame_data = defaultdict(lambda: {
            'gt_instruments': [],
            'gt_actions': [],
            'gt_pairs': [],
            'pred_instruments': [],
            'pred_actions': [],
            'pred_pairs': []
        })

    def accumulate_video_statistics(self):
        """Accumulate current video statistics into total statistics"""
        # Accumulate instrument counts
        for inst in self.instrument_id_to_name.values():
            self.total_gt_instrument_counts[inst] += self.gt_instrument_counts[inst]
            self.total_pred_instrument_counts[inst] += self.pred_instrument_counts[inst]
        
        # Accumulate action counts
        for action in self.action_id_to_name.values():
            self.total_gt_action_counts[action] += self.gt_action_counts[action]
            self.total_pred_action_counts[action] += self.pred_action_counts[action]
        
        # Accumulate instrument-action pairs
        all_pairs = set(list(self.gt_instrument_action_pairs.keys()) + 
                       list(self.pred_instrument_action_pairs.keys()))
        for pair in all_pairs:
            self.total_gt_instrument_action_pairs[pair] += self.gt_instrument_action_pairs[pair]
            self.total_pred_instrument_action_pairs[pair] += self.pred_instrument_action_pairs[pair]

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
                        self.gt_instrument_counts[instrument_name] += 1
                        
                        instrument_info = {
                            'instance_id': instance_id,
                            'name': instrument_name,
                            'bbox': instrument.get('bbox', []),
                            'category_id': category_id
                        }
                        
                        actions = instrument.get('actions', [])
                        if isinstance(actions, list):
                            valid_actions = []
                            for action_id in actions:
                                if action_id in self.action_id_to_name:
                                    action_name = self.action_id_to_name[action_id]
                                    valid_actions.append(action_name)
                                    self.gt_action_counts[action_name] += 1/len(actions)
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

    def evaluate_frame(self, img_path, frame_annotations):
        """Evaluate a single frame with the multitask model"""
        frame_predictions = []
        frame_number = int(os.path.basename(img_path).split('.')[0])
        
        try:
            # Load and process image
            img = Image.open(img_path)
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                iv_output, tool_output, verb_output, _ = self.model(img_tensor)
                
                # Apply sigmoid to get probabilities
                tool_probs = torch.sigmoid(tool_output)[0]
                verb_probs = torch.sigmoid(verb_output)[0]
                
                # Process tool predictions
                for tool_idx, tool_prob in enumerate(tool_probs):
                    if tool_prob >= CONFIDENCE_THRESHOLD and tool_idx in TOOL_MAPPING:
                        cholect50_instrument = TOOL_MAPPING[tool_idx]
                        grasp_instruments = CHOLECT50_TO_GRASP_INSTRUMENT_MAPPING.get(cholect50_instrument)
                        
                        if grasp_instruments:
                            if isinstance(grasp_instruments, str):
                                grasp_instruments = [grasp_instruments]
                                
                            for grasp_instrument in grasp_instruments:
                                # Update instrument predictions
                                self.pred_instrument_counts[grasp_instrument] += 1
                                
                                # Process verb predictions for this instrument
                                for verb_idx, verb_prob in enumerate(verb_probs):
                                    if verb_prob >= CONFIDENCE_THRESHOLD:
                                        cholect50_verb = VERB_MAPPING[verb_idx]
                                        possible_grasp_verbs = CHOLECT50_TO_GRASP_VERB_MAPPING.get(cholect50_verb, [])
                                        
                                        if possible_grasp_verbs:
                                            # Convert to set if it's a single string
                                            if isinstance(possible_grasp_verbs, str):
                                                possible_grasp_verbs = {possible_grasp_verbs}
                                            else:
                                                possible_grasp_verbs = set(possible_grasp_verbs)
                                            
                                            # Get ground truth actions for this instrument
                                            gt_actions = set()
                                            for gt_inst in frame_annotations['gt_instruments']:
                                                if gt_inst['name'] == grasp_instrument and 'valid_actions' in gt_inst:
                                                    gt_actions.update(gt_inst['valid_actions'])
                                            
                                            # Check for overlap between predicted and ground truth
                                            matching_verbs = possible_grasp_verbs & gt_actions
                                            
                                            if matching_verbs:
                                                matched_verb = next(iter(matching_verbs))
                                                self.pred_action_counts[matched_verb] += 1
                                                
                                                pair_key = f"{grasp_instrument}_{matched_verb}"
                                                self.pred_instrument_action_pairs[pair_key] += 1
                                                
                                                prediction = {
                                                    'instrument': {
                                                        'name': grasp_instrument,
                                                        'confidence': float(tool_prob)
                                                    },
                                                    'action': {
                                                        'name': matched_verb,
                                                        'confidence': float(verb_prob)
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

    def prepare_binary_labels(self, category):
        """Prepare binary labels for evaluation"""
        if category == 'instruments':
            all_instruments = sorted(self.instrument_id_to_name.values())
            labels = {
                'gt': [],
                'pred': []
            }
            
            for frame_data in self.frame_data.values():
                gt_instruments = set(inst['name'] for inst in frame_data['gt_instruments'])
                pred_instruments = set(frame_data['pred_instruments'])
                
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
        
        # Calculate and print overall summary metrics
        print("\nOVERALL METRIC SUMMARY:")
        print("-"*50)
        
        categories = ["Instruments", "Actions", "Instrument-Action Pairs"]
        metrics_sets = [
            (total_gt_instruments, total_pred_instruments, instrument_metrics),
            (total_gt_actions, total_pred_actions, action_metrics),
            (total_gt_pairs, total_pred_pairs, pair_metrics)
        ]
        
        for category, (gt_total, pred_total, category_metrics) in zip(categories, metrics_sets):
            mean_precision = sum(m['precision'] for m in category_metrics.values()) / len(category_metrics) if category_metrics else 0
            mean_recall = sum(m['recall'] for m in category_metrics.values()) / len(category_metrics) if category_metrics else 0
            f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0
            
            print(f"\n{category}:")
            print(f"Mean Precision: {mean_precision:.3f}")
            print(f"Mean Recall: {mean_recall:.3f}")
            print(f"F1 Score: {f1_score:.3f}")

    def print_total_statistics(self):
        """Print comprehensive statistics for all processed videos combined"""
        print("\n" + "="*120)
        print("TOTAL STATISTICS ACROSS ALL VIDEOS")
        print("="*120)
        
        # Print instrument statistics
        print("\nINSTRUMENT INSTANCES:")
        print("-"*120)
        print(f"{'Instrument Type':<25} {'GT Count':<10} {'Pred Count':<10} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<10} {'Recall':<10}")
        print("-"*120)
        
        total_gt_instruments = 0
        total_pred_instruments = 0
        total_tp_instruments = 0
        total_fp_instruments = 0
        total_fn_instruments = 0
        
        for instr in sorted(self.instrument_id_to_name.values()):
            gt_count = self.total_gt_instrument_counts[instr]
            pred_count = self.total_pred_instrument_counts[instr]
            
            tp = min(gt_count, pred_count)
            fp = max(0, pred_count - gt_count)
            fn = max(0, gt_count - pred_count)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            total_gt_instruments += gt_count
            total_pred_instruments += pred_count
            total_tp_instruments += tp
            total_fp_instruments += fp
            total_fn_instruments += fn
            
            print(f"{instr:<25} {gt_count:<10} {pred_count:<10} {tp:<8} {fp:<8} {fn:<8} {precision:,.3f}    {recall:,.3f}")
        
        print("-"*120)
        total_precision = total_tp_instruments / (total_tp_instruments + total_fp_instruments) if (total_tp_instruments + total_fp_instruments) > 0 else 0
        total_recall = total_tp_instruments / (total_tp_instruments + total_fn_instruments) if (total_tp_instruments + total_fn_instruments) > 0 else 0
        print(f"{'Total':<25} {total_gt_instruments:<10} {total_pred_instruments:<10} {total_tp_instruments:<8} {total_fp_instruments:<8} {total_fn_instruments:<8} {total_precision:,.3f}    {total_recall:,.3f}")
        
        # Print action statistics
        print("\nACTIONS:")
        print("-"*120)
        print(f"{'Action':<25} {'GT Count':<10} {'Pred Count':<10} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<10} {'Recall':<10}")
        print("-"*120)
        
        total_gt_actions = 0
        total_pred_actions = 0
        total_tp_actions = 0
        total_fp_actions = 0
        total_fn_actions = 0
        
        for action in sorted(self.action_id_to_name.values()):
            gt_count = self.total_gt_action_counts[action]
            pred_count = self.total_pred_action_counts[action]
            
            tp = min(gt_count, pred_count)
            fp = max(0, pred_count - gt_count)
            fn = max(0, gt_count - pred_count)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            total_gt_actions += gt_count
            total_pred_actions += pred_count
            total_tp_actions += tp
            total_fp_actions += fp
            total_fn_actions += fn
            
            print(f"{action:<25} {gt_count:<10.1f} {pred_count:<10} {tp:<8.1f} {fp:<8.1f} {fn:<8.1f} {precision:,.3f}    {recall:,.3f}")
        
        print("-"*120)
        total_precision = total_tp_actions / (total_tp_actions + total_fp_actions) if (total_tp_actions + total_fp_actions) > 0 else 0
        total_recall = total_tp_actions / (total_tp_actions + total_fn_actions) if (total_tp_actions + total_fn_actions) > 0 else 0
        print(f"{'Total':<25} {total_gt_actions:<10.1f} {total_pred_actions:<10} {total_tp_actions:<8.1f} {total_fp_actions:<8.1f} {total_fn_actions:<8.1f} {total_precision:,.3f}    {total_recall:,.3f}")
        
        # Print pair statistics
        print("\nINSTRUMENT-ACTION PAIRS:")
        print("-"*120)
        print(f"{'Pair':<40} {'GT Count':<10} {'Pred Count':<10} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<10} {'Recall':<10}")
        print("-"*120)
        
        total_gt_pairs = 0
        total_pred_pairs = 0
        total_tp_pairs = 0
        total_fp_pairs = 0
        total_fn_pairs = 0
        
        all_pairs = sorted(set(list(self.total_gt_instrument_action_pairs.keys()) + 
                             list(self.total_pred_instrument_action_pairs.keys())))
        
        for pair in all_pairs:
            gt_count = self.total_gt_instrument_action_pairs[pair]
            pred_count = self.total_pred_instrument_action_pairs[pair]
            
            tp = min(gt_count, pred_count)
            fp = max(0, pred_count - gt_count)
            fn = max(0, gt_count - pred_count)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            total_gt_pairs += gt_count
            total_pred_pairs += pred_count
            total_tp_pairs += tp
            total_fp_pairs += fp
            total_fn_pairs += fn
            
            print(f"{pair:<40} {gt_count:<10.1f} {pred_count:<10} {tp:<8.1f} {fp:<8.1f} {fn:<8.1f} {precision:,.3f}    {recall:,.3f}")
        
        print("-"*120)
        total_precision = total_tp_pairs / (total_tp_pairs + total_fp_pairs) if (total_tp_pairs + total_fp_pairs) > 0 else 0
        total_recall = total_tp_pairs / (total_tp_pairs + total_fn_pairs) if (total_tp_pairs + total_fn_pairs) > 0 else 0
        print(f"{'Total':<40} {total_gt_pairs:<10.1f} {total_pred_pairs:<10} {total_tp_pairs:<8.1f} {total_fp_pairs:<8.1f} {total_fn_pairs:<8.1f} {total_precision:,.3f}    {total_recall:,.3f}")

        # Print overall summary metrics
        print("\nOVERALL METRIC SUMMARY ACROSS ALL VIDEOS:")
        print("-"*50)
        
        categories = {
            "Instruments": (total_gt_instruments, total_pred_instruments, total_tp_instruments, total_fp_instruments, total_fn_instruments),
            "Actions": (total_gt_actions, total_pred_actions, total_tp_actions, total_fp_actions, total_fn_actions),
            "Instrument-Action Pairs": (total_gt_pairs, total_pred_pairs, total_tp_pairs, total_fp_pairs, total_fn_pairs)
        }
        
        for category, (gt_total, pred_total, tp, fp, fn) in categories.items():
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n{category}:")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1 Score: {f1_score:.3f}")

def calculate_advanced_metrics(self):
        """Calculate advanced metrics including per-class AP and mAP"""
        metrics = {
            'instruments': self.prepare_binary_labels('instruments'),
            'actions': self.prepare_binary_labels('actions'),
            'pairs': self.prepare_binary_labels('pairs')
        }
        
        advanced_metrics = {}
        
        for category, labels in metrics.items():
            y_true = np.array(labels['gt'])
            y_pred = np.array(labels['pred'])
            
            # Calculate metrics using sklearn
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            ap = average_precision_score(y_true, y_pred, average='macro')
            
            advanced_metrics[category] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'ap': ap
            }
        
        return advanced_metrics

def aggregate_metrics_across_videos(self, all_metrics):
        """
        Aggregate metrics across multiple videos
        
        Args:
            all_metrics (dict): Dictionary of metrics for each video
        
        Returns:
            dict: Aggregated metrics across all videos
        """
        aggregated = {
            'instruments': {'gt': [], 'pred': []},
            'actions': {'gt': [], 'pred': []},
            'pairs': {'gt': [], 'pred': []}
        }
        
        # Combine metrics from all videos
        for video_metrics in all_metrics.values():
            for category in ['instruments', 'actions', 'pairs']:
                aggregated[category]['gt'].extend(video_metrics[f'{category}_metrics']['gt'])
                aggregated[category]['pred'].extend(video_metrics[f'{category}_metrics']['pred'])
        
        # Calculate final metrics
        final_metrics = {}
        for category in aggregated:
            y_true = np.array(aggregated[category]['gt'])
            y_pred = np.array(aggregated[category]['pred'])
            
            final_metrics[category] = {
                'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'ap': average_precision_score(y_true, y_pred, average='macro')
            }
        
        return final_metrics

def generate_video_report(self, video_id):
        """Generate a detailed report for a specific video"""
        print(f"\n{'='*40} Video Report: {video_id} {'='*40}")
        
        # Print frame-wise statistics
        print("\nFrame-wise Statistics:")
        total_frames = len(self.frame_data)
        frames_with_instruments = sum(1 for frame in self.frame_data.values() if frame['gt_instruments'])
        frames_with_actions = sum(1 for frame in self.frame_data.values() if frame['gt_actions'])
        
        print(f"Total Frames: {total_frames}")
        print(f"Frames with Instruments: {frames_with_instruments} ({frames_with_instruments/total_frames*100:.1f}%)")
        print(f"Frames with Actions: {frames_with_actions} ({frames_with_actions/total_frames*100:.1f}%)")
        
        # Calculate and print advanced metrics
        advanced_metrics = self.calculate_advanced_metrics()
        
        print("\nAdvanced Metrics:")
        for category, metrics in advanced_metrics.items():
            print(f"\n{category.upper()}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
        
        return {
            'frame_stats': {
                'total_frames': total_frames,
                'frames_with_instruments': frames_with_instruments,
                'frames_with_actions': frames_with_actions
            },
            'advanced_metrics': advanced_metrics
        }
def main():
    try:
        print("\nStarting evaluation...")
        
        # Get base directory and set up paths
        base_dir = Path(__file__).resolve().parent.parent.parent.parent  # Navigate up to project root
        
        # Configuration with resolved paths - specific to multitask model
        model_path = base_dir / "models" / "multitask-surgical-workflow" / "checkpoints" / "clean-dust-36" / "clean-dust-36-epoch-epoch=26.ckpt"
        
        # Dataset path for GraSP
        dataset_dir = Path("/data/Bartscht/GrasP/test")
        
        print(f"\nEvaluating model: {model_path}")
        print(f"Dataset path: {dataset_dir}")
        
        # Initialize evaluator
        evaluator = GraSPMultitaskEvaluator(
            model_path=model_path,
            dataset_dir=dataset_dir
        )
        
        # Process videos
        video_ids = ["VID41", "VID47", "VID50", "VID51", "VID53"]  # GraSP specific videos
        all_metrics = {}
        
        for video_id in video_ids:
            print(f"\nProcessing video: {video_id}")
            
            # Load ground truth first
            print("\nLoading ground truth annotations...")
            if not evaluator.load_ground_truth(video_id):
                raise Exception(f"Failed to load ground truth annotations for {video_id}")
            
            # Set up frames directory
            frames_dir = dataset_dir / "Videos" / video_id
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
                        
                        # Process frame
                        predictions = evaluator.evaluate_frame(
                            img_path=str(frame_file),
                            frame_annotations=frame_annotations
                        )
                        
                    except Exception as e:
                        print(f"\nWarning: Error processing frame {frame_num}: {str(e)}")
                        continue
                        
                    pbar.update(1)
            
            # Print statistics for each video
            print("\nGenerating statistics for video:", video_id)
            evaluator.print_statistics()
            
            # Generate detailed report
            video_report = evaluator.generate_video_report(video_id)
            
            # Store metrics for this video
            all_metrics[video_id] = {
                'instrument_metrics': evaluator.prepare_binary_labels('instruments'),
                'action_metrics': evaluator.prepare_binary_labels('actions'),
                'pair_metrics': evaluator.prepare_binary_labels('pairs'),
                'report': video_report
            }

            # Accumulate statistics for the total
            evaluator.accumulate_video_statistics()
        
        # Print total statistics across all videos
        print("\n" + "="*50)
        print("TOTAL STATISTICS ACROSS ALL VIDEOS")
        print("="*50)
        evaluator.print_total_statistics()
        
        # Calculate combined metrics across all videos
        print("\nCalculating combined metrics across all videos...")
        final_metrics = evaluator.aggregate_metrics_across_videos(all_metrics)
        
        # Print final combined metrics
        print("\nFINAL COMBINED METRICS ACROSS ALL VIDEOS:")
        print("-"*50)
        for category, metrics in final_metrics.items():
            print(f"\n{category.upper()} METRICS:")
            for metric_name, value in metrics.items():
                print(f"{metric_name.capitalize()}: {value:.4f}")
        
        print("\nEvaluation complete!")
        return True
        
    except Exception as e:
        print(f"\nError in evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\nStarting GraSP evaluation with multitask model...")
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")