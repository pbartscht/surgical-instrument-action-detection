#!/usr/bin/env python3
"""
Instrument Predictor for Active Learning
Loads trained YOLO model and generates predictions for multiple instruments
and exports them directly to CVAT-compatible ZIP format
"""

import os
from pathlib import Path
import yaml
from ultralytics import YOLO
import cv2
import logging
import argparse
import zipfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_project_root():
    """Returns project root folder."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))

def parse_args():
    parser = argparse.ArgumentParser(description='Active Learning Instrument Predictor')
    parser.add_argument('--config', type=str,
                       default=get_default_config_path(),
                       help='Path to active learning config file')
    return parser.parse_args()

def get_default_config_path():
    """Returns the default path to the active learning config file."""
    project_root = get_project_root()
    return os.path.join(project_root, 'config', 'active_learning', 'instrument_config.yaml')

class InstrumentPredictor:
    def __init__(self, config_path: str):
        """
        Initialize the predictor.
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                logging.info(f"Loaded configuration from: {config_path}")
        except FileNotFoundError:
            logging.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing config file: {e}")
            raise

        # Get absolute paths
        project_root = get_project_root()
        model_path = os.path.join(project_root, self.config['paths']['weights'])

        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            logging.info(f"Model loaded from: {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

        # Set up paths
        self.project_root = project_root
        self.predictions_dir = os.path.join(project_root, self.config['paths']['predictions'])
        os.makedirs(self.predictions_dir, exist_ok=True)

    def create_cvat_export(self):
        """Generate predictions and create CVAT ZIP file without temporary files."""
        base_dir = self.config['dataset']['base_dir']
        instruments = self.config['target_instruments']
        
        # Create ZIP file
        zip_path = os.path.join(self.predictions_dir, "cvat_annotations.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Write obj.names
            obj_names = '\n'.join(instr['name'] for instr in instruments)
            zipf.writestr("obj.names", obj_names)
            
            # Write obj.data
            obj_data_content = f"classes = {len(instruments)}\nnames = obj.names\ntrain = train.txt\n"
            zipf.writestr("obj.data", obj_data_content)
            
            # Collect all train.txt entries
            train_txt_entries = []
            
            # Process all frames
            for video in self.config['dataset']['videos']:
                video_dir = Path(base_dir) / 'videos' / video
                if not video_dir.exists():
                    logging.warning(f"Video directory not found: {video_dir}")
                    continue

                logging.info(f"Processing video: {video}")
                total_predictions = 0

                # Process each frame
                for frame_path in sorted(video_dir.glob('*.png')):
                    # Load image for dimensions only
                    img = cv2.imread(str(frame_path))
                    if img is None:
                        logging.warning(f"Failed to load image: {frame_path}")
                        continue

                    # YOLO prediction
                    results = self.model(img)[0]

                    # Process predictions for all instruments
                    all_predictions = []
                    frame_predictions = 0
                    for box in results.boxes:
                        cls_idx = int(box.cls)
                        conf = float(box.conf)
                        
                        # Find matching instrument configuration
                        matching_instrument = next(
                            (instr for instr in instruments if instr['index'] == cls_idx),
                            None
                        )
                        
                        if matching_instrument and conf >= matching_instrument['confidence_threshold']:
                            # Convert to YOLO format
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            w = x2 - x1
                            h = y2 - y1
                            x_center = x1 + w/2
                            y_center = y1 + h/2

                            # Normalize coordinates
                            img_w, img_h = img.shape[1], img.shape[0]
                            x_center /= img_w
                            y_center /= img_h
                            w /= img_w
                            h /= img_h

                            all_predictions.append(f"{cls_idx} {x_center} {y_center} {w} {h}")
                            frame_predictions += 1

                    # Add entry to train.txt
                    train_txt_entries.append(f"obj_train_data/{frame_path.stem}.png")
                    
                    # Always write txt file, even if empty
                    label_content = '\n'.join(all_predictions) if all_predictions else ""
                    zipf.writestr(f"obj_train_data/{frame_path.stem}.txt", label_content)
                    
                    if frame_predictions > 0:
                        total_predictions += frame_predictions
                
                logging.info(f"Video {video}: Processed with {total_predictions} total predictions")

            # Write train.txt
            train_txt_content = '\n'.join(train_txt_entries)
            zipf.writestr("train.txt", train_txt_content)
        
        logging.info(f"CVAT-compatible annotations ZIP file created: {zip_path}")

def main():
    args = parse_args()
    predictor = InstrumentPredictor(args.config)
    predictor.create_cvat_export()

if __name__ == "__main__":
    main()