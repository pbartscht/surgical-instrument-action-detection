import os
from collections import defaultdict
from ultralytics import YOLO
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import csv

CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.3
VIDEOS_TO_ANALYZE = ['VID92']

def yolo_to_instrument_name(yolo_class):
    mapping = {
        0: "grasper",
        1: "bipolar",
        2: "hook",
        3: "scissors",
        4: "clipper",
        5: "irrigator",
        6: "specimenBag"
    }
    return mapping[yolo_class]

def load_ground_truth(txt_file):
    ground_truth = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    current_video = ""
    current_frame = ""
    with open(txt_file, 'r') as f:
        for line in f:
            if line.startswith("Video:"):
                current_video = line.split(":")[1].strip()
            elif line.startswith("Frame:"):
                current_frame = int(line.split(":")[1].strip())
            elif line.strip().startswith("Instrument"):
                instrument = line.split(":")[1].strip()
                ground_truth[current_video][current_frame][instrument] += 1
    return ground_truth

def non_max_suppression(boxes, scores, iou_threshold):
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    keep = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    return keep.numpy()

def evaluate_model(model, dataset_dir, ground_truth, conf_threshold, iou_threshold):
    videos_folder = os.path.join(dataset_dir, "CholecT50", "videos")
    all_predictions = defaultdict(lambda: defaultdict(list))
    all_ground_truths = defaultdict(lambda: defaultdict(int))
    
    for video in VIDEOS_TO_ANALYZE:
        video_folder = os.path.join(videos_folder, video)
        frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.png')])
        
        for frame_file in tqdm(frame_files, desc=f"Processing {video}"):
            frame_number = int(frame_file.split('.')[0])
            img_path = os.path.join(video_folder, frame_file)
            
            img = Image.open(img_path)
            results = model(img)
            
            pred_classes = results[0].boxes.cls.cpu().numpy()
            pred_confidences = results[0].boxes.conf.cpu().numpy()
            pred_boxes = results[0].boxes.xyxy.cpu().numpy()
            
            frame_predictions = defaultdict(list)
            frame_ground_truths = ground_truth[video][frame_number]
            
            for instrument_class in set(pred_classes):
                class_mask = pred_classes == instrument_class
                class_boxes = pred_boxes[class_mask]
                class_confidences = pred_confidences[class_mask]
                
                if len(class_boxes) > 0:
                    keep = non_max_suppression(class_boxes, class_confidences, iou_threshold)
                    
                    instrument = yolo_to_instrument_name(int(instrument_class))
                    for idx in keep:
                        if class_confidences[idx] >= conf_threshold:
                            frame_predictions[instrument].append({
                                'confidence': class_confidences[idx],
                                'box': class_boxes[idx]
                            })
            
            for instrument in set(list(frame_predictions.keys()) + list(frame_ground_truths.keys())):
                all_predictions[instrument][f"{video}_{frame_number}"] = frame_predictions[instrument]
                all_ground_truths[instrument][f"{video}_{frame_number}"] = frame_ground_truths[instrument]
    
    return all_predictions, all_ground_truths

def calculate_custom_ap(y_true, y_scores):
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true = np.array(y_true)[sorted_indices]
    y_scores = np.array(y_scores)[sorted_indices]
    
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    
    precision = tp / (tp + fp)
    recall = tp / np.sum(y_true)
    
    # Calculate AP using the formula: AP = sum((r_n - r_{n-1}) * p_n)
    ap = 0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i-1]) * precision[i]
    
    return ap

def calculate_mAP(all_predictions, all_ground_truths):
    APs = {}
    for instrument in all_predictions.keys():
        y_true = []
        y_scores = []
        for frame in all_predictions[instrument].keys():
            gt_count = all_ground_truths[instrument][frame]
            pred_confidences = [pred['confidence'] for pred in all_predictions[instrument][frame]]
            
            y_true.extend([1] * gt_count + [0] * len(pred_confidences))
            y_scores.extend(pred_confidences + [0] * gt_count)
        
        if len(y_true) > 0 and sum(y_true) > 0:
            AP = calculate_custom_ap(y_true, y_scores)
            APs[instrument] = AP
            print(f"AP for {instrument}: {AP:.4f}")
        else:
            print(f"Skipping AP calculation for {instrument} (no occurrences)")
    
    mAP = sum(APs.values()) / len(APs) if APs else 0
    print(f"Mean Average Precision (mAP): {mAP:.4f}")
    
    return APs, mAP

def generate_csv(all_predictions, all_ground_truths, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Frame', 'Instrument', 'Ground Truth', 'Prediction'])
        
        for frame in sorted(set(frame for instrument in all_ground_truths for frame in all_ground_truths[instrument])):
            video, frame_number = frame.split('_')
            for instrument in set(list(all_predictions.keys()) + list(all_ground_truths.keys())):
                gt_count = all_ground_truths[instrument][frame]
                pred_count = len(all_predictions[instrument][frame])
                if gt_count > 0 or pred_count > 0:
                    csvwriter.writerow([frame_number, instrument, gt_count, pred_count])

# Usage
model = YOLO('/home/Bartscht/YOLO/runs/detect/train11/weights/best.pt')
dataset_dir = "/data/Bartscht"
ground_truth_file = "/home/Bartscht/YOLO/improve_YOLO_Dataset/test_cholect50_annotations.txt"
output_csv = "evaluation_results.csv"

print("Loading ground truth data...")
ground_truth = load_ground_truth(ground_truth_file)

print("Evaluating model...")
all_predictions, all_ground_truths = evaluate_model(model, dataset_dir, ground_truth, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

print("Generating CSV file...")
generate_csv(all_predictions, all_ground_truths, output_csv)
print(f"CSV file generated: {output_csv}")

print("Calculating mAP...")
APs, mAP = calculate_mAP(all_predictions, all_ground_truths)

print("\nAverage Precision for each instrument:")
for instrument, AP in APs.items():
    print(f"{instrument}: {AP:.4f}")

print(f"\nMean Average Precision (mAP): {mAP:.4f}")