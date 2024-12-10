import csv
import numpy as np
from collections import defaultdict

def load_csv(file_path):
    data = defaultdict(list)
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            instrument = row['Instrument']
            ground_truth = int(row['Ground Truth'])
            prediction = int(row['Prediction'])
            data[instrument].append((ground_truth, prediction))
    return data

def calculate_precision_recall(ground_truth, prediction):
    tp = min(ground_truth, prediction)
    fp = max(0, prediction - ground_truth)
    fn = max(0, ground_truth - prediction)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall

def calculate_ap(instrument_data):
    precisions = []
    recalls = []
    total_gt = sum(gt for gt, _ in instrument_data)
    
    if total_gt == 0:
        return 0  # No positive samples, AP is 0
    
    running_tp = 0
    running_fp = 0
    
    for gt, pred in sorted(instrument_data, key=lambda x: -x[1]):  # Sort by prediction count, descending
        tp = min(gt, pred)
        fp = max(0, pred - gt)
        running_tp += tp
        running_fp += fp
        
        precision = running_tp / (running_tp + running_fp) if (running_tp + running_fp) > 0 else 0
        recall = running_tp / total_gt
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AP using the trapezoidal rule
    ap = 0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i-1]) * precisions[i]
    
    return ap

def calculate_map(data):
    aps = {}
    for instrument, instrument_data in data.items():
        ap = calculate_ap(instrument_data)
        aps[instrument] = ap
        total_gt = sum(gt for gt, _ in instrument_data)
        total_pred = sum(pred for _, pred in instrument_data)
        tp = sum(min(gt, pred) for gt, pred in instrument_data)
        fp = sum(max(0, pred - gt) for gt, pred in instrument_data)
        fn = sum(max(0, gt - pred) for gt, pred in instrument_data)
        
        print(f"{instrument}:")
        print(f"  AP: {ap:.4f}")
        print(f"  True Positives: {tp}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  Total Ground Truth instances: {total_gt}")
        print(f"  Total Predicted instances: {total_pred}")
        print(f"  Total frames: {len(instrument_data)}")
        print()
    
    mAP = sum(aps.values()) / len(aps) if aps else 0
    print(f"Mean Average Precision (mAP): {mAP:.4f}")
    
    return aps, mAP

# Usage
csv_file = "evaluation_results.csv"
data = load_csv(csv_file)
aps, mAP = calculate_map(data)