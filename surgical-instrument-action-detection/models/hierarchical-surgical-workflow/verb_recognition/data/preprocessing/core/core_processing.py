import os
from collections import defaultdict
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import torch
import numpy as np
import json
import cv2
from tqdm import tqdm

from config import (
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD, OUTPUT_SIZE,
    VIDEOS_TO_PROCESS, INSTRUMENT_MAPPING, VERB_MAPPING
)

def yolo_to_instrument_name(yolo_class):
    return INSTRUMENT_MAPPING[yolo_class]

def non_max_suppression(boxes, scores, iou_threshold):
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    keep = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    return keep.numpy()

def create_directory_structure(base_dir):
    """Create the required directory structure"""
    labels_dir = os.path.join(base_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    video_paths = {}
    for video in VIDEOS_TO_PROCESS:
        video_path = os.path.join(base_dir, video)
        os.makedirs(video_path, exist_ok=True)
        video_paths[video] = video_path
    
    return labels_dir, video_paths

def load_ground_truth_from_json(dataset_dir, video):
    """
    Verbesserte Ground Truth Ladung mit Überprüfung auf eindeutige Verb-Zuordnungen
    """
    labels_folder = os.path.join(dataset_dir, "CholecT50", "labels")
    json_file = os.path.join(labels_folder, f"{video}.json")
    
    frame_annotations = defaultdict(lambda: {
        'instruments': defaultdict(int),
        'verbs': defaultdict(set)  # Nutze set für unique Verben
    })
    
    with open(json_file, 'r') as f:
        data = json.load(f)
        annotations = data['annotations']
        
        # Sammle erst alle Informationen
        for frame, instances in annotations.items():
            frame_number = int(frame)
            for instance in instances:
                instrument = instance[1]
                verb = instance[7]
                
                if isinstance(instrument, int) and 0 <= instrument < 6:
                    instrument_name = yolo_to_instrument_name(instrument)
                    frame_annotations[frame_number]['instruments'][instrument_name] += 1
                    
                    if isinstance(verb, int) and 0 <= verb < 10:
                        verb_name = VERB_MAPPING[verb]
                        frame_annotations[frame_number]['verbs'][instrument_name].add(verb_name)
        
        # Bereinige nicht eindeutige Fälle
        final_annotations = defaultdict(dict)
        for frame_number, frame_data in frame_annotations.items():
            final_annotations[frame_number]['instruments'] = {}
            final_annotations[frame_number]['verbs'] = {}
            
            for instrument, count in frame_data['instruments'].items():
                verbs = frame_data['verbs'][instrument]
                # Nur wenn genau eine Instanz und ein eindeutiges Verb
                if count == 1 and len(verbs) == 1:
                    final_annotations[frame_number]['instruments'][instrument] = 1
                    final_annotations[frame_number]['verbs'][instrument] = verbs.pop()
    
    return final_annotations, data

def get_predictions_for_frame(results, confidence_threshold, iou_threshold):
    """Process YOLO predictions for a single frame"""
    try:
        pred_classes = results[0].boxes.cls.cpu().numpy()
        pred_confidences = results[0].boxes.conf.cpu().numpy()
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()
        
        valid_mask = (pred_classes < 6) & (pred_confidences >= confidence_threshold)
        pred_classes = pred_classes[valid_mask]
        pred_confidences = pred_confidences[valid_mask]
        pred_boxes = pred_boxes[valid_mask]
        
        final_predictions = {}
        
        for instrument_class in set(pred_classes):
            class_mask = pred_classes == instrument_class
            class_boxes = pred_boxes[class_mask]
            class_confidences = pred_confidences[class_mask]
            
            if len(class_boxes) > 0:
                keep = non_max_suppression(class_boxes, class_confidences, iou_threshold)
                instrument = yolo_to_instrument_name(int(instrument_class))
                
                if len(keep) == 1:
                    final_predictions[instrument] = {
                        'box': class_boxes[keep[0]],
                        'confidence': class_confidences[keep[0]]
                    }
        
        return final_predictions
    except Exception as e:
        print(f"Error in frame prediction: {str(e)}")
        return {}

def process_video(model, dataset_dir, video_paths, labels_dir, video_name):
    """Process a single video"""
    try:
        print(f"Processing {video_name}")
        
        # Load ground truth and save to new location
        ground_truth, original_data = load_ground_truth_from_json(dataset_dir, video_name)
        
        # Save ground truth JSON to new labels directory
        json_output_path = os.path.join(labels_dir, f"{video_name}.json")
        with open(json_output_path, 'w') as f:
            json.dump(original_data, f, indent=4)
        
        # Process frames
        video_folder = os.path.join(dataset_dir, "CholecT50", "videos", video_name)
        label_data = []
        
        frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.png')])
        
        for frame_file in tqdm(frame_files, desc=f"Processing {video_name}"):
            frame_number = int(frame_file.split('.')[0])
            img_path = os.path.join(video_folder, frame_file)
            
            frame_gt = ground_truth[frame_number]
            
            # YOLO Prediction
            img_pil = Image.open(img_path)
            results = model(img_pil)
            final_predictions = get_predictions_for_frame(results, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)
            
            # Process predictions
            for instrument, pred_data in final_predictions.items():
                if (instrument in frame_gt.get('instruments', {}) and 
                    instrument in frame_gt.get('verbs', {})):
                    
                    img_cv2 = cv2.imread(img_path)
                    box = pred_data['box'].astype(int)
                    conf = pred_data['confidence']
                    
                    cropped = img_cv2[box[1]:box[3], box[0]:box[2]]
                    if cropped.size > 0:
                        resized = cv2.resize(cropped, OUTPUT_SIZE)
                        verb = frame_gt['verbs'][instrument]
                        
                        # Save image in video directory
                        output_filename = f"{frame_number:04d}_{instrument}_{verb}_conf{conf:.2f}.png"
                        output_path = os.path.join(video_paths[video_name], output_filename)
                        cv2.imwrite(output_path, resized)
                        
                        # Add to labels
                        label_data.append({
                            'Dateiname': output_filename,
                            'Verb': verb,
                            'Instrument': instrument,
                            'Frame': frame_number,
                            'Confidence': conf
                        })
        
        # Save labels
        if label_data:
            df = pd.DataFrame(label_data)
            csv_path = os.path.join(video_paths[video_name], 'labels.csv')
            df.to_csv(csv_path, index=False)
            
            print(f"\nCompleted {video_name}: {len(df)} crops generated")
            print("\nInstrument-Verb pair distribution:")
            print(df.groupby(['Instrument', 'Verb']).size())
            
            return len(df)
        return 0
    except Exception as e:
        print(f"Error processing video {video_name}: {str(e)}")
        return 0

def process_all_videos(model, dataset_dir, verbs_dir):
    """Process all specified videos"""
    # Create directory structure
    labels_dir, video_paths = create_directory_structure(verbs_dir)
    
    print(f"Starting processing {len(VIDEOS_TO_PROCESS)} videos...")
    
    total_crops = 0
    for video in VIDEOS_TO_PROCESS:
        try:
            crops_count = process_video(model, dataset_dir, video_paths, labels_dir, video)
            total_crops += crops_count
        except Exception as e:
            print(f"Failed to process {video}: {str(e)}")
    
    print(f"\nProcessing complete. Total crops generated: {total_crops}")

