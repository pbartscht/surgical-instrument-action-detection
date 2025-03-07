import os
import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
import json
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO

# Path configuration
current_dir = Path(__file__).resolve().parent
hierarchical_dir = current_dir.parent
sys.path.append(str(hierarchical_dir))

# Constants
CONFIDENCE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.3
VIDEOS_TO_ANALYZE = ["VID92", "VID96", "VID103", "VID110", "VID111"]

# Global mappings
TOOL_MAPPING = {
    0: 'grasper', 1: 'bipolar', 2: 'hook', 
    3: 'scissors', 4: 'clipper', 5: 'irrigator'
}

class ModelLoader:
    def __init__(self):
        self.hierarchical_dir = hierarchical_dir
        self.setup_paths()

    def setup_paths(self):
        """Defines all important paths for the models"""
        # YOLO model path
        #self.yolo_weights = self.hierarchical_dir / "Instrument-classification-detection/weights/instrument_detector/weights/epoch90.pt"
        self.yolo_weights = self.hierarchical_dir / "Instrument-classification-detection/weights/instrument_detector/epoch100.pt"
        # Dataset path
        self.dataset_path = Path("/data/Bartscht/CholecT50")
        
        print(f"YOLO weights path: {self.yolo_weights}")
        print(f"Dataset path: {self.dataset_path}")

        # Validate paths
        if not self.yolo_weights.exists():
            raise FileNotFoundError(f"YOLO weights not found at: {self.yolo_weights}")
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
        
class InstrumentEvaluator:
    def __init__(self, yolo_model, dataset_dir):
        """
        Initializes the InstrumentEvaluator.
        
        :param yolo_model: Pre-trained YOLO model for instrument detection
        :param dataset_dir: Directory of the CholecT50 dataset
        """
        # YOLO model for instrument detection
        self.yolo_model = yolo_model
        
        # Path to dataset
        self.dataset_dir = dataset_dir

    def load_ground_truth(self, video):
        """
        Loads ground truth annotations for a specific video.
        
        :param video: Video identifier (e.g., "VID92")
        :return: Dictionary with frame annotations
        """
        # Path to label files
        labels_folder = os.path.join(self.dataset_dir, "labels")
        json_file = os.path.join(labels_folder, f"{video}.json")
        
        # Defaultdict for frame annotations
        frame_annotations = defaultdict(lambda: {
            'instruments': defaultdict(int)
        })
        
        # Load JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)
            annotations = data['annotations']
            
            # Process annotations
            for frame, instances in annotations.items():
                frame_number = int(frame)
                for instance in instances:
                    instrument = instance[1]
                    
                    # Validate and map instrument
                    if isinstance(instrument, int) and 0 <= instrument < 6:
                        instrument_name = TOOL_MAPPING[instrument]
                        frame_annotations[frame_number]['instruments'][instrument_name] += 1
        
        return frame_annotations

    def evaluate_frame(self, img_path, ground_truth, save_visualization=False):
        """
        Evaluates a single frame using instrument detection
        
        Args:
            img_path: Path to the frame image
            ground_truth: Ground truth annotations for this frame
            save_visualization: Whether to save visualization of detections
        
        Returns:
            List of predictions, each containing:
            [video_frame, instrument, instrument_conf]
        """
        # Initialize predictions list and extract frame information
        frame_predictions = []
        frame_number = int(os.path.basename(img_path).split('.')[0])
        video_name = os.path.basename(os.path.dirname(img_path))
        
        # Load and prepare image for visualization
        img = Image.open(img_path)
        original_img = img.copy() if save_visualization else None
        draw = ImageDraw.Draw(original_img) if save_visualization else None
        
        try:
            if save_visualization:
                # Try to load a nice font, fallback to default if not available
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                    small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                except:
                    font = ImageFont.load_default()
                    small_font = ImageFont.load_default()
        
            # Instrument Detection using YOLO with verbose=False to suppress output
            yolo_results = self.yolo_model(img, verbose=False)
            valid_detections = []
            
            # Process each YOLO detection
            for detection in yolo_results[0].boxes:
                instrument_class = int(detection.cls)
                confidence = float(detection.conf)
                
                # Only consider predictions above confidence threshold
                if instrument_class < 6 and confidence >= CONFIDENCE_THRESHOLD:
                    instrument_name = TOOL_MAPPING[instrument_class]
                    
                    # Store valid detection
                    valid_detections.append({
                        'class': instrument_class,
                        'confidence': confidence,
                        'box': detection.xyxy[0],
                        'name': instrument_name
                    })
            
            # Sort detections by confidence (process high confidence detections first)
            valid_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Process each detected instrument
            for idx, detection in enumerate(valid_detections):
                instrument_name = detection['name']
                box = detection['box']
                confidence = detection['confidence']
                
                # Format: [frame_id, instrument, instrument_conf]
                prediction = [
                    f"{video_name}_frame{frame_number}",
                    instrument_name,
                    confidence
                ]
                frame_predictions.append(prediction)
                
                # Visualization: Draw bounding box and predictions (if needed)
                if save_visualization:
                    x1, y1, x2, y2 = map(int, box)
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                    text_color = 'blue' if confidence >= CONFIDENCE_THRESHOLD else 'orange'
                    draw.text((x1, y1-25), 
                            f"Pred: {instrument_name}\n"
                            f"Conf: {confidence:.2f}", 
                            fill=text_color, font=font)
            
            # Add ground truth visualization (if needed)
            if save_visualization:
                img_width, img_height = original_img.size
                gt_y_start = img_height - 100  # Start position for ground truth text
                
                # Draw background for ground truth
                draw.rectangle([10, gt_y_start, img_width-10, img_height-10], 
                            fill='white', outline='black')
                
                # Write ground truth heading and instruments
                draw.text((20, gt_y_start + 5), "Ground Truth:", fill='black', font=font)
                
                y_pos = gt_y_start + 35
                for instrument, count in ground_truth['instruments'].items():
                    if count > 0:
                        draw.text((20, y_pos), f"GT: {instrument}", fill='green', font=small_font)
                        y_pos += 20
                
                # Save visualization
                viz_dir = os.path.join("/data/Bartscht/VID92_val", "visualizations")
                os.makedirs(viz_dir, exist_ok=True)
                save_path = os.path.join(viz_dir, f"{video_name}_frame{frame_number}.png")
                original_img.save(save_path)
            
            return frame_predictions
                
        except Exception as e:
            print(f"Error processing frame {video_name}_{frame_number}: {str(e)}")
            return []

    def evaluate(self):
        """
        Comprehensive evaluation of model performance across all specified videos.
        
        The evaluation process:
        1. Initializes tracking for ALL possible instrument classes
        2. Processes each video frame by frame
        3. Updates metrics and predictions
        4. Calculates final metrics including:
            - Traditional metrics (Precision, Recall, F1)
            - Average Precision (AP) for ALL classes
            - mean Average Precision (mAP) across all classes
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Initialize collection structures for ALL possible instrument classes
        all_predictions = {
            'instruments': {instr: [] for instr in TOOL_MAPPING.values()}
        }
        
        # For each instrument, maintain a list of all ground truths and predictions across all videos
        all_gt_pred_pairs = {
            'instruments': {instr: {'gt': [], 'pred': [], 'conf': []} for instr in TOOL_MAPPING.values()}
        }
        
        # Track occurrences of each instrument in ground truth across all videos
        ground_truth_occurrences = {
            'instruments': defaultdict(int)
        }
        
        # Per-video statistics for logging and debugging
        video_stats = {}
        
        # Overall statistics across all videos
        total_stats = {
            'TP': 0, 'FP': 0, 'FN': 0,
            'frames_processed': 0,
            'instruments_detected': 0,
            'ground_truth_instruments': 0
        }
        
        # Process each video in the evaluation set
        for video in VIDEOS_TO_ANALYZE:
            # Initialize per-video counters
            video_stats[video] = {
                'TP': 0, 'FP': 0, 'FN': 0, 
                'frames_processed': 0,
                'instruments_detected': 0,
                'ground_truth_instruments': 0,
                'per_instrument': {instr: {'TP': 0, 'FP': 0, 'FN': 0, 'gt_count': 0, 'pred_count': 0} 
                                 for instr in TOOL_MAPPING.values()}
            }
            
            # Load ground truth annotations
            ground_truth = self.load_ground_truth(video)
            
            # Get frame files
            video_folder = os.path.join(self.dataset_dir, "videos", video)
            frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.png')])
            
            # Process each frame
            for frame_file in tqdm(frame_files, desc=f"Evaluating {video}"):
                frame_number = int(frame_file.split('.')[0])
                img_path = os.path.join(video_folder, frame_file)
                
                # Get frame predictions
                frame_predictions = self.evaluate_frame(
                    img_path,
                    ground_truth[frame_number],
                    save_visualization=(video == "VID92" and frame_number % 100 == 0)  # Save occasional frames
                )
                
                # Get ground truth for this frame
                gt_frame = ground_truth[frame_number]
                
                # Update frame counters
                video_stats[video]['frames_processed'] += 1
                total_stats['frames_processed'] += 1
                
                # Count instruments in ground truth for this frame and update counters
                frame_gt_instruments = {}
                frame_gt_count = 0
                
                for instr, count in gt_frame['instruments'].items():
                    if count > 0:
                        frame_gt_instruments[instr] = count
                        frame_gt_count += count
                        
                        # Update instrument-specific counters
                        video_stats[video]['per_instrument'][instr]['gt_count'] += count
                        
                        # Update global ground truth occurrences
                        ground_truth_occurrences['instruments'][instr] += count
                
                # Update video and total ground truth counters
                video_stats[video]['ground_truth_instruments'] += frame_gt_count
                total_stats['ground_truth_instruments'] += frame_gt_count
                
                # Count predictions for this frame
                video_stats[video]['instruments_detected'] += len(frame_predictions)
                total_stats['instruments_detected'] += len(frame_predictions)
                
                # Create dictionaries for matched instruments to track TP/FP/FN
                matched_instruments = set()
                
                # Track predictions for each instrument in this frame
                predicted_instruments = defaultdict(int)
                
                # Process frame predictions
                for pred in frame_predictions:
                    _, instrument, instrument_conf = pred
                    
                    # Update prediction counts
                    predicted_instruments[instrument] += 1
                    video_stats[video]['per_instrument'][instrument]['pred_count'] += 1
                    
                    # Add to all_predictions for later AP calculation
                    # Create binary ground truth indicator
                    gt_present = 1 if instrument in frame_gt_instruments else 0
                    
                    # Store prediction details
                    all_predictions['instruments'][instrument].append({
                        'confidence': instrument_conf,
                        'ground_truth': gt_present,
                        'frame': frame_number,
                        'video': video
                    })
                    
                    # Store in the consolidated GT-pred pairs for this instrument
                    all_gt_pred_pairs['instruments'][instrument]['gt'].append(gt_present)
                    all_gt_pred_pairs['instruments'][instrument]['pred'].append(1)  # Predicted as present
                    all_gt_pred_pairs['instruments'][instrument]['conf'].append(instrument_conf)
                    
                    # Update metrics based on ground truth
                    if gt_present:  # True Positive
                        video_stats[video]['TP'] += 1
                        total_stats['TP'] += 1
                        video_stats[video]['per_instrument'][instrument]['TP'] += 1
                        matched_instruments.add(instrument)
                    else:  # False Positive
                        video_stats[video]['FP'] += 1
                        total_stats['FP'] += 1
                        video_stats[video]['per_instrument'][instrument]['FP'] += 1
                
                # Add entries for ground truth instruments that were not detected (FN)
                for instr, count in frame_gt_instruments.items():
                    if instr not in matched_instruments:
                        # Count as false negative
                        video_stats[video]['FN'] += count
                        total_stats['FN'] += count
                        video_stats[video]['per_instrument'][instr]['FN'] += count
                        
                        # Also add to the consolidated GT-pred pairs
                        for _ in range(count):
                            all_gt_pred_pairs['instruments'][instr]['gt'].append(1)  # Ground truth: present
                            all_gt_pred_pairs['instruments'][instr]['pred'].append(0)  # Prediction: absent
                            all_gt_pred_pairs['instruments'][instr]['conf'].append(0.0)  # No confidence for absence
                
                # For instruments not in ground truth and not predicted, add entries for completeness
                for instr in TOOL_MAPPING.values():
                    if instr not in frame_gt_instruments and instr not in predicted_instruments:
                        # This is a True Negative case
                        all_gt_pred_pairs['instruments'][instr]['gt'].append(0)  # Ground truth: absent
                        all_gt_pred_pairs['instruments'][instr]['pred'].append(0)  # Prediction: absent
                        all_gt_pred_pairs['instruments'][instr]['conf'].append(0.0)  # No confidence
        
        # Print per-video statistics
        print("\nPer-Video Statistics:")
        print("-" * 80)
        print(f"{'Video':<10} {'Frames':<8} {'GT Insts':<10} {'Detections':<12} {'TP':<6} {'FP':<6} {'FN':<6}")
        print("-" * 80)
        
        for video, stats in video_stats.items():
            print(f"{video:<10} {stats['frames_processed']:<8} {stats['ground_truth_instruments']:<10} "
                  f"{stats['instruments_detected']:<12} {stats['TP']:<6} {stats['FP']:<6} {stats['FN']:<6}")
        
        print("-" * 80)
        print(f"Total: {total_stats['frames_processed']} frames, {total_stats['ground_truth_instruments']} "
              f"ground truth instruments, {total_stats['instruments_detected']} detections")
        print(f"Overall TP: {total_stats['TP']}, FP: {total_stats['FP']}, "
              f"FN: {total_stats['FN']}")
        
        # Calculate and format final metrics
        final_metrics = self._calculate_metrics(all_gt_pred_pairs, ground_truth_occurrences, total_stats)
        
        # Add video statistics to final metrics
        final_metrics['video_stats'] = video_stats
        final_metrics['total_stats'] = total_stats
        
        # Print detailed evaluation results
        self._print_evaluation_results(final_metrics)
        
        return final_metrics

    def _calculate_metrics(self, all_gt_pred_pairs, ground_truth_occurrences, total_stats):
        """
        Calculate all evaluation metrics from the collected data.
        
        Args:
            all_gt_pred_pairs: Dictionary with all ground truth and prediction pairs
            ground_truth_occurrences: Count of ground truth occurrences
            total_stats: Overall TP, FP, FN counts
            
        Returns:
            Dictionary with all calculated metrics
        """
        # Initialize results dictionary
        results = {
            'instruments': {
                'per_class': {},
                'mean': {},
                'weighted_mean': {},
                'traditional': {}
            }
        }
        
        # Calculate traditional overall metrics
        try:
            TP = total_stats['TP']
            FP = total_stats['FP']
            FN = total_stats['FN']
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results['instruments']['traditional'] = {
                'TP': TP,
                'FP': FP,
                'FN': FN,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        except Exception as e:
            print(f"Error calculating traditional metrics: {e}")
            results['instruments']['traditional'] = {
                'TP': 0, 'FP': 0, 'FN': 0,
                'precision': 0, 'recall': 0, 'f1': 0
            }
        
        # Calculate per-instrument metrics
        category = 'instruments'
        valid_classes = []
        total_support = sum(ground_truth_occurrences[category].values())
        
        for instrument, data in all_gt_pred_pairs[category].items():
            # Skip if no data for this instrument
            if not data['gt']:
                results[category]['per_class'][instrument] = {
                    'AP': 0.0,
                    'support': 0,
                    'dist': 0,
                    'pred': 0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'TP': 0,
                    'FP': 0,
                    'FN': 0
                }
                continue
            
            # Get ground truth and prediction arrays
            y_true = np.array(data['gt'])
            y_pred = np.array(data['pred'])
            y_scores = np.array(data['conf'])
            
            # Calculate support (ground truth occurrences)
            support = ground_truth_occurrences[category][instrument]
            
            # Calculate TP, FP, FN
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            # Calculate metrics
            try:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                # Calculate AP only if needed (when we have positive samples)
                if support > 0 and np.sum(y_scores > 0) > 0:
                    ap = average_precision_score(y_true, y_scores)
                else:
                    ap = 0.0
            except Exception as e:
                print(f"Error calculating metrics for {instrument}: {e}")
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                ap = 0.0
            
            # Store results
            results[category]['per_class'][instrument] = {
                'AP': ap,
                'support': support,
                'dist': (support / total_support * 100) if total_support > 0 else 0,
                'pred': np.sum(y_pred),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'TP': int(tp),
                'FP': int(fp),
                'FN': int(fn)
            }
            
            # Add to valid classes for mean calculation if instrument appears in ground truth
            if support > 0:
                valid_classes.append(results[category]['per_class'][instrument])
        
        # Calculate mean metrics
        if valid_classes:
            results[category]['mean'] = {
                'AP': np.mean([c['AP'] for c in valid_classes]),
                'precision': np.mean([c['precision'] for c in valid_classes]),
                'recall': np.mean([c['recall'] for c in valid_classes]),
                'f1': np.mean([c['f1'] for c in valid_classes])
            }
            
            # Calculate weighted mean metrics
            weights = [c['support'] for c in valid_classes]
            results[category]['weighted_mean'] = {
                'AP': np.average([c['AP'] for c in valid_classes], weights=weights),
                'precision': np.average([c['precision'] for c in valid_classes], weights=weights),
                'recall': np.average([c['recall'] for c in valid_classes], weights=weights),
                'f1': np.average([c['f1'] for c in valid_classes], weights=weights)
            }
        else:
            # Default values if no valid classes
            results[category]['mean'] = {
                'AP': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0
            }
            results[category]['weighted_mean'] = {
                'AP': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0
            }
        
        return results

    def _print_evaluation_results(self, final_metrics):
        """
        Print detailed evaluation results.
        
        Args:
            final_metrics: Dictionary with all calculated metrics
        """
        print("\nInstrument Evaluation Results:")
        print("="*100)
        
        category = 'instruments'
        
        # Print traditional metrics header
        print("\nTraditional Overall Metrics:")
        print(f"Total TP: {final_metrics[category]['traditional']['TP']}, "
              f"Total FP: {final_metrics[category]['traditional']['FP']}, "
              f"Total FN: {final_metrics[category]['traditional']['FN']}")
        print(f"Overall Precision: {final_metrics[category]['traditional']['precision']:.4f}, "
              f"Overall Recall: {final_metrics[category]['traditional']['recall']:.4f}, "
              f"Overall F1: {final_metrics[category]['traditional']['f1']:.4f}")
        
        # Print per-class metrics header
        print("\nPer-Instrument Metrics:")
        print("-"*100)
        print(f"{'Instrument':<15} {'Support':>8} {'Dist%':>8} {'Pred':>8} {'TP':>6} {'FP':>6} {'FN':>6} "
              f"{'AP':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
        print("-"*100)
        
        # Print metrics for each instrument class
        for class_name, metrics in sorted(final_metrics[category]['per_class'].items()):
            print(f"{class_name:<15} {metrics['support']:>8d} {metrics['dist']:>8.2f} {metrics['pred']:>8d} "
                  f"{metrics['TP']:>6d} {metrics['FP']:>6d} {metrics['FN']:>6d} "
                  f"{metrics['AP']:>8.4f} {metrics['f1']:>8.4f} {metrics['precision']:>8.4f} {metrics['recall']:>8.4f}")
        
        # Print mean metrics
        print("-"*100)
        print(f"Mean         {'-':>8} {'-':>8} {'-':>8} {'-':>6} {'-':>6} {'-':>6} "
              f"{final_metrics[category]['mean']['AP']:>8.4f} "
              f"{final_metrics[category]['mean']['f1']:>8.4f} "
              f"{final_metrics[category]['mean']['precision']:>8.4f} "
              f"{final_metrics[category]['mean']['recall']:>8.4f}")
        
        # Print weighted mean metrics
        print(f"Weighted Mean{'-':>8} {'-':>8} {'-':>8} {'-':>6} {'-':>6} {'-':>6} "
              f"{final_metrics[category]['weighted_mean']['AP']:>8.4f} "
              f"{final_metrics[category]['weighted_mean']['f1']:>8.4f} "
              f"{final_metrics[category]['weighted_mean']['precision']:>8.4f} "
              f"{final_metrics[category]['weighted_mean']['recall']:>8.4f}")

def main():
    # Initialize ModelLoader
    try:
        loader = ModelLoader()
        
        # Load models
        yolo_model = loader.load_yolo_model()
        
        # Dataset directory
        dataset_dir = str(loader.dataset_path)
        
        # Verify dataset structure
        labels_dir = os.path.join(dataset_dir, "labels")
        videos_dir = os.path.join(dataset_dir, "videos")
        
        print("\nDataset Structure Check:")
        print(f"Labels Directory: {labels_dir}")
        print(f"Videos Directory: {videos_dir}")
        
        # Verify directories exist
        if not os.path.exists(labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        if not os.path.exists(videos_dir):
            raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
        
        # List available videos
        available_videos = [v for v in os.listdir(videos_dir) 
                           if os.path.isdir(os.path.join(videos_dir, v))]
        
        # Check that all videos to analyze are available
        missing_videos = [v for v in VIDEOS_TO_ANALYZE if v not in available_videos]
        if missing_videos:
            print(f"Warning: The following videos are not available in the dataset: {missing_videos}")
            print(f"Available videos: {available_videos}")
        
        print("\nAvailable Videos:")
        for video in available_videos:
            if video in VIDEOS_TO_ANALYZE:
                print(f"- {video} (selected for evaluation)")
            else:
                print(f"- {video}")
        
        # Create Evaluator
        evaluator = InstrumentEvaluator(
            yolo_model=yolo_model,
            dataset_dir=dataset_dir
        )
        
        # Run Evaluation
        results = evaluator.evaluate()
        
        print("\nInstrument Evaluation Completed Successfully!")

         # Optional: Save results to file
        try:
            output_dir = os.path.join(current_dir, "evaluation_results")
            os.makedirs(output_dir, exist_ok=True)
            
            import pickle
            with open(os.path.join(output_dir, "instrument_eval_results.pkl"), "wb") as f:
                pickle.dump(results, f)
            print(f"Results saved to {output_dir}/instrument_eval_results.pkl")
            
            # Save metrics summary as CSV
            csv_summary = []
            csv_summary.append("Instrument,Support,DistPct,Pred,TP,FP,FN,AP,F1,Precision,Recall")
            
            category = 'instruments'
            for class_name, metrics in sorted(results[category]['per_class'].items()):
                csv_summary.append(f"{class_name},{metrics['support']},{metrics['dist']:.2f},"
                                 f"{metrics['pred']},{metrics['TP']},{metrics['FP']},{metrics['FN']},"
                                 f"{metrics['AP']:.4f},{metrics['f1']:.4f},"
                                 f"{metrics['precision']:.4f},{metrics['recall']:.4f}")
# Add traditional overall metrics
            csv_summary.append(f"Overall,,,,"
                            f"{results[category]['traditional']['TP']},"
                            f"{results[category]['traditional']['FP']},"
                            f"{results[category]['traditional']['FN']},"
                            f",{results[category]['traditional']['f1']:.4f},"
                            f"{results[category]['traditional']['precision']:.4f},"
                            f"{results[category]['traditional']['recall']:.4f}")
            
            # Add mean and weighted mean
            csv_summary.append(f"Mean,,,,,,,{results[category]['mean']['AP']:.4f},"
                            f"{results[category]['mean']['f1']:.4f},"
                            f"{results[category]['mean']['precision']:.4f},"
                            f"{results[category]['mean']['recall']:.4f}")
            
            csv_summary.append(f"WeightedMean,,,,,,,{results[category]['weighted_mean']['AP']:.4f},"
                            f"{results[category]['weighted_mean']['f1']:.4f},"
                            f"{results[category]['weighted_mean']['precision']:.4f},"
                            f"{results[category]['weighted_mean']['recall']:.4f}")
            
            with open(os.path.join(output_dir, "instrument_eval_metrics.csv"), "w") as f:
                f.write("\n".join(csv_summary))
            print(f"Metrics saved to {output_dir}/instrument_eval_metrics.csv")
            
            # Additionally save per-video statistics
            video_csv = []
            video_csv.append("Video,Frames,GT_Instruments,Detections,TP,FP,FN")
            
            for video, stats in results['video_stats'].items():
                video_csv.append(f"{video},{stats['frames_processed']},"
                              f"{stats['ground_truth_instruments']},"
                              f"{stats['instruments_detected']},"
                              f"{stats['TP']},{stats['FP']},{stats['FN']}")
            
            with open(os.path.join(output_dir, "per_video_stats.csv"), "w") as f:
                f.write("\n".join(video_csv))
            print(f"Per-video statistics saved to {output_dir}/per_video_stats.csv")
            
        except Exception as e:
            print(f"Warning: Could not save results to file: {str(e)}")
        
    except Exception as e:
        print(f"❌ Error during initialization or evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()