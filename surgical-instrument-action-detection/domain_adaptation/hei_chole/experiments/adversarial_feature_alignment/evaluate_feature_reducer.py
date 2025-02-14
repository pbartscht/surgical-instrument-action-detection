import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
import json
import cv2
import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, average_precision_score

class AdaptedYOLO(nn.Module):
    def __init__(self, yolo_path, feature_reducer_path=None):
        super().__init__()
        # Load YOLO strictly in inference mode
        self.yolo = YOLO(yolo_path)
        self.yolo.model.eval()  # Set to evaluation mode
        self.yolo_model = self.yolo.model.model
        
        # Important: Set task to prediction only
        self.yolo.task = 'predict'
        self.yolo.mode = 'predict'
        
        # Ensure no training can happen
        for param in self.yolo_model.parameters():
            param.requires_grad = False
        self.feature_layer = 8
        
        # Disable gradients for YOLO model
        for param in self.yolo_model.parameters():
            param.requires_grad = False
        
        # Load feature reducer if provided
        self.feature_reducer = None
        if feature_reducer_path:
            checkpoint = torch.load(feature_reducer_path)
            self.feature_reducer = nn.Sequential(
                nn.Conv2d(512, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.5)
            )
            self.feature_reducer.load_state_dict(checkpoint['feature_reducer'])
            self.feature_reducer.eval()

    def predict(self, img, use_adapter=True):
        """
        Predict using either original YOLO or adapted YOLO
        """
        with torch.no_grad():  # Ensure no gradients are computed
            return self.yolo.predict(
                source=img,
                conf=0.1,      # Confidence threshold
                iou=0.3,       # NMS IOU threshold
                verbose=False,
                mode='predict',  # Explicitly set predict mode
                imgsz=640       # Use standard image size
            )

def load_ground_truth(video, dataset_dir):
    """
    Loads binary ground truth annotations for HeiChole dataset.
    """
    labels_folder = os.path.join(dataset_dir, "Labels")
    json_file = os.path.join(labels_folder, f"{video}.json")
    
    frame_annotations = defaultdict(lambda: {
        'instruments': defaultdict(int),
        'actions': defaultdict(int)
    })
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            frames = data.get('frames', {})
            
            for frame_num, frame_data in frames.items():
                frame_number = int(frame_num)
                instruments = frame_data.get('instruments', {})
                for instr_name, present in instruments.items():
                    frame_annotations[frame_number]['instruments'][instr_name] = 1 if present > 0 else 0
            
            return frame_annotations
                
    except Exception as e:
        print(f"Error loading annotations: {str(e)}")
        raise

def evaluate_instruments(predictions, ground_truth, instrument_classes):
    """
    Calculate metrics for instrument classification
    """
    metrics = {}
    
    y_true = []
    y_pred = []
    y_score = []
    
    for instr in instrument_classes:
        for frame_num in ground_truth.keys():
            true_value = ground_truth[frame_num]['instruments'].get(instr, 0)
            pred_value = predictions[frame_num]['instruments'].get(instr, 0)
            score_value = predictions[frame_num].get('scores', {}).get(instr, 0)
            
            y_true.append(true_value)
            y_pred.append(pred_value)
            y_score.append(score_value)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_score, average='macro')
        
        metrics[instr] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'ap': ap
        }
    
    return metrics

def process_video(video_path, model, use_adapter=True):
    """
    Process video frames and return predictions
    """
    # Mappings
    TOOL_MAPPING = {
        0: 'grasper', 1: 'bipolar', 2: 'hook',
        3: 'scissors', 4: 'clipper', 5: 'irrigator'
    }
    
    CHOLECT50_TO_HEICHOLE_MAPPING = {
        'grasper': 'grasper',
        'bipolar': 'coagulation',
        'clipper': 'clipper',
        'hook': 'coagulation',
        'scissors': 'scissors',
        'irrigator': 'suction_irrigation'
    }
    
    cap = cv2.VideoCapture(str(video_path))
    frame_predictions = defaultdict(lambda: {
        'instruments': defaultdict(int),
        'scores': defaultdict(float)
    })
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get predictions
        results = model.predict(frame, use_adapter=use_adapter)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                
                if cls_id in TOOL_MAPPING:
                    source_class = TOOL_MAPPING[cls_id]
                    target_class = CHOLECT50_TO_HEICHOLE_MAPPING[source_class]
                    
                    frame_predictions[frame_number]['instruments'][target_class] = 1
                    frame_predictions[frame_number]['scores'][target_class] = max(
                        conf,
                        frame_predictions[frame_number]['scores'][target_class]
                    )
        
        frame_number += 1
    
    cap.release()
    return frame_predictions

def evaluate_model(model_path, target_data_path, feature_reducer_path=None):
    """
    Evaluate YOLO model with and without domain adaptation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with feature reducer
    model = AdaptedYOLO(model_path, feature_reducer_path).to(device)
    model.eval()
    
    # Target domain instrument classes
    instrument_classes = ['grasper', 'coagulation', 'clipper', 'scissors', 'suction_irrigation']
    
    # Get list of test videos
    video_dir = Path(target_data_path) / "Videos"
    videos = list(video_dir.glob("*.mp4"))
    
    if not videos:
        print(f"No videos found in {video_dir}")
        return
    
    results = {
        'with_adapter': {},
        'without_adapter': {}
    }
    
    # Process each video
    for video_path in tqdm(videos, desc="Processing videos"):
        video_name = video_path.stem
        print(f"\nProcessing video: {video_name}")
        
        # Load ground truth
        ground_truth = load_ground_truth(video_name, target_data_path)
        
        # Test with domain adaptation
        print("Testing with domain adaptation...")
        predictions_with = process_video(video_path, model, use_adapter=True)
        metrics_with = evaluate_instruments(predictions_with, ground_truth, instrument_classes)
        results['with_adapter'][video_name] = metrics_with
        
        # Test without domain adaptation
        print("Testing without domain adaptation...")
        predictions_without = process_video(video_path, model, use_adapter=False)
        metrics_without = evaluate_instruments(predictions_without, ground_truth, instrument_classes)
        results['without_adapter'][video_name] = metrics_without
        
        # Print intermediate results
        print(f"\nResults for {video_name}:")
        print("\nWith domain adaptation:")
        for instr, metrics in metrics_with.items():
            print(f"{instr}: F1={metrics['f1']:.3f}, AP={metrics['ap']:.3f}")
        print("\nWithout domain adaptation:")
        for instr, metrics in metrics_without.items():
            print(f"{instr}: F1={metrics['f1']:.3f}, AP={metrics['ap']:.3f}")
    
    # Save results
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "evaluation_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    model_path = "/data/Bartscht/YOLO/best_v35.pt"
    target_data_path = "/data/Bartscht/HeiChole/domain_adaptation/test"
    feature_reducer_path = str(Path.home() / "YOLO/surgical-instrument-action-detection/domain_adaptation/hei_chole/experiments/domain_adapter_weights/best_feature_reducer.pt")
    
    print("Starting evaluation...")
    print(f"Model path: {model_path}")
    print(f"Target data path: {target_data_path}")
    print(f"Feature reducer path: {feature_reducer_path}")
    
    results = evaluate_model(model_path, target_data_path, feature_reducer_path)
    
    print("\nEvaluation completed!")
    print("Results have been saved to evaluation_results/evaluation_metrics.json")