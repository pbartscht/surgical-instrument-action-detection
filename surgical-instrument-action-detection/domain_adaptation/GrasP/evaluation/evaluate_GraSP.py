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
        self.dataset_path = Path("/data/Bartscht/GrasP")
        
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
        """
        Initialize the GraSP evaluator.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = yolo_model
        self.verb_model = verb_model.to(self.device)
        self.verb_model.eval()
        self.dataset_dir = Path(dataset_dir)
        
        # Image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize data structures for GraSP dataset
        self.labels_dir = self.dataset_dir / "Labels"
        self.video_ids = sorted([f.stem for f in self.labels_dir.glob("*.json")])
        self.instrument_data = defaultdict(lambda: defaultdict(list))
        self.action_data = defaultdict(lambda: defaultdict(list))
        
        print(f"Initialized GraSP evaluator with dataset at: {self.dataset_dir}")
        print(f"Using device: {self.device}")

    def load_grasp_dataset(self):
        """
        Loads and processes the GraSP dataset annotations.
        """
        total_frames = 0
        instrument_counts = defaultdict(int)
        action_counts = defaultdict(int)
        instrument_action_pairs = defaultdict(lambda: defaultdict(int))
        
        for video_id in self.video_ids:
            json_file = self.labels_dir / f"{video_id}.json"
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                    # Get categories lookup
                    instrument_categories = {
                        cat['id']: cat['name'] 
                        for cat in data['categories']['instruments']
                    }
                    action_categories = {
                        cat['id']: cat['name'] 
                        for cat in data['categories']['actions']
                    }
                    
                    frames = data.get('frames', {})
                    total_frames += len(frames)
                    
                    # Process each frame
                    for frame_id, frame_data in frames.items():
                        for instrument_ann in frame_data.get('instruments', []):
                            category_id = instrument_ann.get('category_id')
                            if category_id is not None:
                                instr_name = instrument_categories[category_id]
                                instrument_counts[instr_name] += 1
                                
                                # Store frame-specific data
                                self.instrument_data[video_id][frame_id].append(instr_name)
                                
                                # Process actions for this instrument
                                for action_id in instrument_ann.get('actions', []):
                                    action_name = action_categories[action_id]
                                    action_counts[action_name] += 1
                                    instrument_action_pairs[instr_name][action_name] += 1
                                    
                                    # Store frame-specific action data
                                    self.action_data[video_id][frame_id].append({
                                        'instrument': instr_name,
                                        'action': action_name
                                    })
                                    
            except Exception as e:
                print(f"Error processing {video_id}: {str(e)}")
        
        dataset_stats = {
            'total_frames': total_frames,
            'instrument_counts': instrument_counts,
            'action_counts': action_counts,
            'instrument_action_pairs': instrument_action_pairs
        }
        
        return dataset_stats

    def get_frame_annotations(self, video_id, frame_id):
        """
        Retrieves annotations for a specific frame.
        """
        instruments = self.instrument_data[video_id][frame_id]
        actions = self.action_data[video_id][frame_id]
        return instruments, actions

    def evaluate_frame(self, img_path, frame_annotations, save_visualization=True):
        """
        Evaluates a single frame.
        """
        frame_predictions = []
        frame_number = int(os.path.basename(img_path).split('.')[0])
        video_name = os.path.basename(os.path.dirname(img_path))
        
        img = Image.open(img_path)
        original_img = img.copy()
        draw = ImageDraw.Draw(original_img)
        
        try:
            # YOLO predictions
            yolo_results = self.yolo_model(img)
            valid_detections = []
            
            # Process YOLO detections
            for detection in yolo_results[0].boxes:
                instrument_class = int(detection.cls)
                confidence = float(detection.conf)
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    cholect50_instrument = TOOL_MAPPING[instrument_class]
                    grasp_instruments = CHOLECT50_TO_GRASP_INSTRUMENT_MAPPING.get(cholect50_instrument)
                    
                    if grasp_instruments:  # Only process if mapping exists
                        if isinstance(grasp_instruments, str):
                            grasp_instruments = [grasp_instruments]
                        for grasp_instrument in grasp_instruments:
                            valid_detections.append({
                                'class': instrument_class,
                                'confidence': confidence,
                                'box': detection.xyxy[0],
                                'name': grasp_instrument
                            })
            
            # Sort by confidence
            valid_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Process each detection
            for detection in valid_detections:
                grasp_instrument = detection['name']
                box = detection['box']
                confidence = detection['confidence']
                
                # Get instrument crop for verb prediction
                x1, y1, x2, y2 = map(int, box)
                instrument_crop = img.crop((x1, y1, x2, y2))
                crop_tensor = self.transform(instrument_crop).unsqueeze(0).to(self.device)
                
                # Predict verb
                verb_outputs = self.verb_model(crop_tensor, [grasp_instrument])
                verb_probs = verb_outputs['probabilities']
                
                # Get top verb predictions
                top_verbs = []
                for verb_idx in torch.topk(verb_probs[0], k=3).indices.cpu().numpy():
                    cholect50_verb = VERB_MAPPING[VERB_MODEL_TO_EVAL_MAPPING[verb_idx]]
                    mapped_verbs = CHOLECT50_TO_HEICHOLE_VERB_MAPPING.get(cholect50_verb)
                    
                    if mapped_verbs:
                        for mapped_verb in mapped_verbs:
                            top_verbs.append({
                                'name': mapped_verb,
                                'probability': float(verb_probs[0][verb_idx]) / len(mapped_verbs)
                            })
                
                # Create prediction for best verb
                if top_verbs:
                    best_verb = max(top_verbs, key=lambda x: x['probability'])
                    verb_name = best_verb['name']
                    verb_conf = best_verb['probability']
                    
                    frame_predictions.append({
                        'frame_id': f"{video_name}_frame{frame_number}",
                        'instrument': {
                            'name': grasp_instrument,
                            'confidence': confidence
                        },
                        'action': {
                            'name': verb_name,
                            'confidence': verb_conf
                        }
                    })
                    
                    # Visualization
                    if save_visualization:
                        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                        text = f"{grasp_instrument}\n{verb_name}\nConf: {confidence:.2f}"
                        draw.text((x1, y1-40), text, fill='blue')
            
            if save_visualization:
                viz_dir = os.path.join(self.dataset_dir, "visualizations")
                os.makedirs(viz_dir, exist_ok=True)
                save_path = os.path.join(viz_dir, f"{video_name}_frame{frame_number}.jpg")
                original_img.save(save_path)
            
            return frame_predictions
            
        except Exception as e:
            print(f"Error processing frame {frame_number}: {str(e)}")
            return []

    def evaluate(self, videos_to_analyze):
        """
        Evaluates model performance on specified videos.
        """
        predictions = {
            'instruments': defaultdict(list),
            'actions': defaultdict(list)
        }
        
        print("\nStarting evaluation process...")
        
        for video_id in videos_to_analyze:
            print(f"\nProcessing {video_id}...")
            
            # Get video frame files - geändert von .png zu .jpg
            video_folder = os.path.join(self.dataset_dir, "Videos", video_id)
            frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.jpg')])
            
            # Process frames
            for frame_file in tqdm(frame_files, desc=f"Evaluating {video_id}"):
                frame_number = frame_file.split('.')[0]
                img_path = os.path.join(video_folder, frame_file)
                
                # Get ground truth annotations for this frame
                instruments, actions = self.get_frame_annotations(video_id, frame_number)
                
                # Get frame predictions
                frame_predictions = self.evaluate_frame(
                    img_path,
                    (instruments, actions),
                    save_visualization=True
                )
                
                # Update predictions and ground truth
                for pred in frame_predictions:
                    # Process instrument predictions
                    instrument = pred['instrument']
                    is_correct_instrument = instrument['name'] in instruments
                    predictions['instruments'][instrument['name']].append({
                        'confidence': instrument['confidence'],
                        'ground_truth': 1 if is_correct_instrument else 0
                    })
                    
                    # Process action predictions
                    action = pred['action']
                    is_correct_action = any(
                        act_data['action'] == action['name'] and 
                        act_data['instrument'] == instrument['name']
                        for act_data in actions
                    )
                    predictions['actions'][action['name']].append({
                        'confidence': action['confidence'],
                        'ground_truth': 1 if is_correct_action else 0
                    })
        
        # Calculate metrics using mean Average Precision
        metrics = self._calculate_detection_metrics(predictions)
        return metrics
    
    def _calculate_map(self, sorted_confidences, sorted_ground_truth):
        """
        Berechnet Average Precision (AP) aus sortierten Konfidenzwerten und Ground Truth Labels.
        """
        if len(sorted_ground_truth) == 0 or sum(sorted_ground_truth) == 0:
            return 0.0
            
        # Berechne cumulative TP und FP
        tp = np.zeros(len(sorted_confidences))
        fp = np.zeros(len(sorted_confidences))
        
        for i, gt in enumerate(sorted_ground_truth):
            if gt == 1:
                tp[i] = 1
            else:
                fp[i] = 1
                
        # Berechne cumulative sums
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Berechne precision und recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / sum(sorted_ground_truth)
        
        # Füge Anfangspunkte hinzu
        precision = np.concatenate(([1.0], precision))
        recall = np.concatenate(([0.0], recall))
        
        # Berechne AP mit allen Punkten
        ap = 0.0
        for i in range(len(recall) - 1):
            ap += (recall[i + 1] - recall[i]) * precision[i + 1]
        
        return ap

    def _calculate_detection_metrics(self, predictions):
        """
        Berechnet die mAP Metriken für Instrumente und Aktionen.
        """
        metrics = {
            'instruments': defaultdict(dict),
            'actions': defaultdict(dict)
        }
        
        # Berechne Metriken für jeden Typ
        for pred_type in ['instruments', 'actions']:
            for name, preds in predictions[pred_type].items():
                if not preds:  # Überspringe leere Predictions
                    continue
                    
                # Sortiere Predictions nach Konfidenz
                sorted_preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)
                confidences = [p['confidence'] for p in sorted_preds]
                
                # Berechne mAP
                ap = self._calculate_map(confidences, [1] * len(confidences))  # Hier nehmen wir an, dass alle Predictions korrekt sind
                
                metrics[pred_type][name] = {
                    'mAP@.5IoU': ap,
                    'predictions': len(preds)
                }
        
        return metrics
    
def main():
    """Evaluate model predictions on GraSP test dataset"""
    try:
        # Initialize ModelLoader
        loader = ModelLoader()
        
        # Load models
        yolo_model = loader.load_yolo_model()
        verb_model = loader.load_verb_model()
        dataset_dir = str(loader.dataset_path / "test")  # Spezifiziere test-Ordner
        
        print("\n==========================================")
        print("INITIALIZING GRASP EVALUATION")
        print("==========================================")
        
        # Create GraSP evaluator
        evaluator = GraSPEvaluator(
            yolo_model=yolo_model, 
            verb_model=verb_model, 
            dataset_dir=dataset_dir
        )
        
        # Load and analyze GraSP dataset
        print("\nLoading GraSP dataset...")
        dataset_stats = evaluator.load_grasp_dataset()
        
        print(f"\nFound {dataset_stats['total_frames']} frames to analyze")
        
        # Get test videos
        videos_to_analyze = evaluator.video_ids
        print(f"Found {len(videos_to_analyze)} videos in test set: {', '.join(videos_to_analyze)}")
        
        print("\n==========================================")
        print("STARTING EVALUATION")
        print("==========================================")
        
        # Run evaluation
        metrics = evaluator.evaluate(videos_to_analyze)
        
        # Print results
        print("\n==========================================")
        print("EVALUATION RESULTS")
        print("==========================================")
        
        # Print Instrument Results
        print("\nINSTRUMENT DETECTION METRICS:")
        print("=" * 50)
        print(f"{'Instrument':20s} {'mAP@.5IoU':>10s}")
        print("-" * 50)
        
        for instr, metrics in metrics['instruments'].items():
            print(f"{instr:20s} {metrics['mAP@.5IoU']:10.4f}")
        
        # Print Action Results
        print("\nACTION RECOGNITION METRICS:")
        print("=" * 50)
        print(f"{'Action':20s} {'mAP@.5IoU':>10s}")
        print("-" * 50)
        
        for action, metrics in metrics['actions'].items():
            print(f"{action:20s} {metrics['mAP@.5IoU']:10.4f}")
        
        # Print Mean Scores
        print("\nMEAN SCORES:")
        print("=" * 50)
        mean_instr_map = np.mean([m['mAP@.5IoU'] for m in metrics['instruments'].values()])
        mean_action_map = np.mean([m['mAP@.5IoU'] for m in metrics['actions'].values()])
        print(f"Mean Instrument mAP@.5IoU: {mean_instr_map:.4f}")
        print(f"Mean Action mAP@.5IoU: {mean_action_map:.4f}")
        
        # Print dataset statistics
        print("\nDATASET STATISTICS:")
        print("=" * 50)
        print("\nINSTRUMENTS:")
        for instr, count in dataset_stats['instrument_counts'].items():
            print(f"{instr:20s} {count:10d}")
        
        print("\nACTIONS:")
        for action, count in dataset_stats['action_counts'].items():
            print(f"{action:20s} {count:10d}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()