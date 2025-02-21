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

class IVPairMapper:
    def __init__(self):
        # Mapping für Instrument-Verb-Paare zum IV-Output-Index
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
        
        # Reverse mapping für einfache Suche
        self.pair_to_index = {pair: idx for idx, pair in self.iv_pairs.items()}
        
        # Separate mappings für Instrumente und Verben
        self.instruments = sorted(list(set(pair[0] for pair in self.iv_pairs.values())))
        self.verbs = sorted(list(set(pair[1] for pair in self.iv_pairs.values())))
        
        # Mapping zu GraSP-Format
        self.instrument_to_grasp = {
            'grasper': ['Prograsp Forceps', 'Laparoscopic Grasper'],
            'bipolar': ['Bipolar Forceps'],
            'scissors': ['Monopolar Curved Scissors'],
            'clipper': ['Clip Applier'],
            'irrigator': ['Suction Instrument'],
            'hook': None  # Kein entsprechendes Instrument in GraSP
        }
        
        self.verb_to_grasp = {
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
    def map_prediction_to_grasp(self, iv_idx, confidence):
        """
        Wandelt eine IV-Prediction in GraSP-Format um
        
        Args:
            iv_idx: Index der IV-Vorhersage
            confidence: Konfidenzwert der Vorhersage
            
        Returns:
            List von Dictionaries mit GraSP-Instrument und -Verb Kombinationen
        """
        results = []
        
        # Hole das Instrument-Verb-Paar für den Index
        iv_pair = self.get_iv_pair(iv_idx)
        if not iv_pair:
            return results
            
        cholect_instrument, cholect_verb = iv_pair
        
        # Hole die entsprechenden GraSP-Instrumente und -Verben
        grasp_instruments = self.get_grasp_instruments(cholect_instrument)
        grasp_verbs = self.get_grasp_verbs(cholect_verb)
        
        # Wenn keine Mappings gefunden wurden, return leere Liste
        if not grasp_instruments or not grasp_verbs:
            return results
        
        # Generiere alle möglichen Kombinationen
        for grasp_instrument in grasp_instruments:
            for grasp_verb in grasp_verbs:
                results.append({
                    'instrument': {
                        'name': grasp_instrument,
                        'confidence': float(confidence)
                    },
                    'action': {
                        'name': grasp_verb,
                        'confidence': float(confidence)
                    }
                })
        
        return results

    def get_iv_pair(self, index):
        """Gibt das Instrument-Verb-Paar für einen bestimmten Index zurück"""
        return self.iv_pairs.get(index)
    
    def get_iv_index(self, instrument, verb):
        """Gibt den Index für ein Instrument-Verb-Paar zurück"""
        return self.pair_to_index.get((instrument, verb))
    
    def get_grasp_instruments(self, cholect_instrument):
        """Konvertiert ein CholecT50-Instrument in GraSP-Instrumente"""
        return self.instrument_to_grasp.get(cholect_instrument, [])
    
    def get_grasp_verbs(self, cholect_verb):
        """Konvertiert ein CholecT50-Verb in GraSP-Verben"""
        return self.verb_to_grasp.get(cholect_verb, [])
    
class GraSPMultitaskEvaluator:
    def __init__(self, model_path, dataset_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CholecT50Model.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.dataset_dir = Path(dataset_dir)
        self.iv_mapper = IVPairMapper()

        self.total_gt_instrument_counts = defaultdict(int)
        self.total_pred_instrument_counts = defaultdict(int)
        self.total_gt_action_counts = defaultdict(float)
        self.total_pred_action_counts = defaultdict(int)
        self.total_gt_instrument_action_pairs = defaultdict(float)
        self.total_pred_instrument_action_pairs = defaultdict(int)
        
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

    def evaluate_frame(self, img_path, frame_annotations):
        """Evaluate a single frame using specialized outputs"""
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
                iv_probs = torch.sigmoid(iv_output)[0]
                tool_probs = torch.sigmoid(tool_output)[0]
                verb_probs = torch.sigmoid(verb_output)[0]
                
                # Step 1: Evaluate Instrument Instances using tool_output
                self._evaluate_instruments(tool_probs, frame_annotations)
                
                # Step 2: Evaluate Actions using verb_output
                self._evaluate_actions(verb_probs, frame_annotations)
                
                # Step 3: Evaluate Instrument-Action Pairs using iv_output
                frame_predictions = self._evaluate_pairs(iv_probs, frame_annotations)
                
                # Update frame data
                self._update_frame_data(frame_number, frame_predictions)
                
            return frame_predictions
            
        except Exception as e:
            print(f"Error processing frame {frame_number}: {str(e)}")
            return []

    def _evaluate_instruments(self, tool_probs, frame_annotations):
        """Evaluate instrument instances using tool_output"""
        for tool_idx, tool_prob in enumerate(tool_probs):
            if tool_prob >= CONFIDENCE_THRESHOLD and tool_idx in TOOL_MAPPING:
                cholect50_instrument = TOOL_MAPPING[tool_idx]
                grasp_instruments = self.iv_mapper.get_grasp_instruments(cholect50_instrument)
                
                if not grasp_instruments:
                    continue
                    
                for grasp_instrument in grasp_instruments:
                    # Check if this instrument appears in ground truth
                    gt_has_instrument = any(
                        gt_inst['name'] == grasp_instrument 
                        for gt_inst in frame_annotations['gt_instruments']
                    )
                    
                    if gt_has_instrument:
                        self.pred_instrument_counts[grasp_instrument] += 1

    def _evaluate_actions(self, verb_probs, frame_annotations):
        """Evaluate actions using verb_output"""
        for verb_idx, verb_prob in enumerate(verb_probs):
            if verb_prob >= CONFIDENCE_THRESHOLD and verb_idx in VERB_MAPPING:
                cholect50_verb = VERB_MAPPING[verb_idx]
                grasp_verbs = self.iv_mapper.get_grasp_verbs(cholect50_verb)
                
                if not grasp_verbs:
                    continue
                    
                for grasp_verb in grasp_verbs:
                    # Check if this action appears in ground truth
                    gt_has_action = False
                    for gt_inst in frame_annotations['gt_instruments']:
                        if 'valid_actions' in gt_inst and grasp_verb in gt_inst['valid_actions']:
                            gt_has_action = True
                            break
                    
                    if gt_has_action:
                        self.pred_action_counts[grasp_verb] += 1

    def _evaluate_pairs(self, iv_probs, frame_annotations):
        """Evaluate instrument-action pairs using iv_output"""
        frame_predictions = []
        
        for iv_idx, iv_prob in enumerate(iv_probs):
            if iv_prob >= CONFIDENCE_THRESHOLD:
                # Convert IV prediction to GraSP format
                grasp_predictions = self.iv_mapper.map_prediction_to_grasp(iv_idx, iv_prob)
                
                for pred in grasp_predictions:
                    instrument_name = pred['instrument']['name']
                    action_name = pred['action']['name']
                    
                    # Check if this pair appears in ground truth
                    gt_has_pair = False
                    for gt_inst in frame_annotations['gt_instruments']:
                        if (gt_inst['name'] == instrument_name and 
                            'valid_actions' in gt_inst and 
                            action_name in gt_inst['valid_actions']):
                            gt_has_pair = True
                            break
                    
                    if gt_has_pair:
                        pair_key = f"{instrument_name}_{action_name}"
                        self.pred_instrument_action_pairs[pair_key] += 1
                        frame_predictions.append(pred)
        
        return frame_predictions

    def print_instrument_statistics(self):
        """Print statistics for instruments based on tool_output"""
        print("\nINSTRUMENT INSTANCES (Based on tool_output):")
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
            tp = min(gt_count, pred_count)
            fp = max(0, pred_count - gt_count)
            fn = max(0, gt_count - pred_count)
            
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

    def print_action_statistics(self):
        """Print statistics for actions based on verb_output"""
        print("\nACTIONS (Based on verb_output):")
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

    def print_pair_statistics(self):
        """Print statistics for instrument-action pairs based on iv_output"""
        print("\nINSTRUMENT-ACTION PAIRS (Based on iv_output):")
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

    def print_total_instrument_statistics(self):
        """Print total instrument statistics across all videos"""
        print("\nINSTRUMENT INSTANCES (Based on tool_output):")
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

    def print_total_action_statistics(self):
        """Print total action statistics across all videos"""
        print("\nACTIONS (Based on verb_output):")
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

    def print_total_pair_statistics(self):
        """Print total instrument-action pair statistics across all videos"""
        print("\nINSTRUMENT-ACTION PAIRS (Based on iv_output):")
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
        

    def _update_frame_data(self, frame_number, frame_predictions):
        """Update frame data with predictions"""
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

def main():
    try:
        print("\nStarting evaluation...")
        
        # Get base directory and set up paths
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        
        # Configuration with resolved paths - specific to multitask model
        model_path = base_dir / "models" / "multitask-surgical-workflow" / "checkpoints" / "clean-dust-36" / "clean-dust-36-epoch-epoch=26.ckpt"
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
            
            # Process each frame with specialized evaluation
            with tqdm(total=len(frame_files)) as pbar:
                for frame_file in frame_files:
                    frame_num = int(frame_file.stem)
                    
                    try:
                        # Get annotations for current frame
                        frame_annotations = evaluator.get_frame_annotations(frame_num)
                        
                        # Process frame using specialized outputs
                        predictions = evaluator.evaluate_frame(
                            img_path=str(frame_file),
                            frame_annotations=frame_annotations
                        )
                        
                    except Exception as e:
                        print(f"\nWarning: Error processing frame {frame_num}: {str(e)}")
                        continue
                        
                    pbar.update(1)
            
            # Generate and print statistics for this video
            print(f"\nGenerating statistics for video: {video_id}")
            print("\n" + "="*120)
            print(f"VIDEO STATISTICS: {video_id}")
            print("="*120)
            
            # Print statistics using tool_output for instruments
            evaluator.print_instrument_statistics()
            
            # Print statistics using verb_output for actions
            evaluator.print_action_statistics()
            
            # Print statistics using iv_output for pairs
            evaluator.print_pair_statistics()
            
            # Generate detailed report for this video
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
        
        # After processing all videos, print total statistics
        print("\n" + "="*120)
        print("TOTAL STATISTICS ACROSS ALL VIDEOS")
        print("="*120)
        
        # Print total statistics using accumulated data
        print("\nINSTRUMENT STATISTICS (Based on tool_output):")
        evaluator.print_total_instrument_statistics()
        
        print("\nACTION STATISTICS (Based on verb_output):")
        evaluator.print_total_action_statistics()
        
        print("\nINSTRUMENT-ACTION PAIR STATISTICS (Based on iv_output):")
        evaluator.print_total_pair_statistics()
        
        # Calculate and print final combined metrics
        print("\nCalculating combined metrics across all videos...")
        final_metrics = evaluator.aggregate_metrics_across_videos(all_metrics)
        
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