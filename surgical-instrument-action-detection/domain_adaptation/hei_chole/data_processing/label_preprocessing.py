#!/usr/bin/env python3
"""
HeiChole Label Processing Script
===============================

This script combines and processes label files from the HeiChole dataset by:
1. Reading individual CSV files for phases, instruments, and actions
2. Combining them into a single JSON structure
3. Sampling labels at specified intervals to match extracted frames
4. Providing consistent label mapping and structure

The script handles the following label types:
- Surgical phases (7 classes)
- Instrument presence (7 instruments)
- Surgical actions (4 types)
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Union
import logging

class HeiCholeLabelProcessor:
    """Processes and combines HeiChole dataset labels."""
    
    PHASE_MAPPING = {
        0: "Preparation",
        1: "Calot triangle dissection",
        2: "Clipping and cutting",
        3: "Gallbladder dissection",
        4: "Gallbladder packaging",
        5: "Cleaning and coagulation",
        6: "Gallbladder retraction"
    }
    
    INSTRUMENTS = [
        "grasper",
        "clipper",
        "coagulation",
        "scissors",
        "suction_irrigation",
        "specimen_bag",
        "stapler"
    ]
    
    ACTIONS = [
        "grasp",
        "hold",
        "cut",
        "clip"
    ]
    
    def __init__(self, base_path: str, output_dir: str = "combined_labels"):
        """
        Initialize the label processor.
        
        Args:
            base_path: Base directory containing the HeiChole dataset
            output_dir: Directory for combined label files
        """
        self.base_path = Path(base_path)
        self.output_dir = self.base_path / output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def read_label_file(self, video_id: str, label_type: str) -> pd.DataFrame:
        """
        Read a single label file.
        
        Args:
            video_id: ID of the video
            label_type: Type of label (Phase/Instrument/Action)
            
        Returns:
            DataFrame containing the labels
        """
        file_path = self.base_path / f'HeiChole_{label_type}labels' / f'{video_id}_Annotation_{label_type}.csv'
        try:
            return pd.read_csv(file_path, header=None)
        except Exception as e:
            logging.error(f"Error reading {label_type} file for {video_id}: {str(e)}")
            raise

    def create_sampled_labels(self, video_id: str, sample_interval: int) -> Dict:
        """
        Create combined and sampled labels for a video.
        
        Args:
            video_id: ID of the video
            sample_interval: Interval for sampling frames
            
        Returns:
            Dictionary containing combined labels
        """
        logging.info(f"Processing labels for {video_id}")
        
        # Read all label files
        phase_df = self.read_label_file(video_id, "Phase")
        instrument_df = self.read_label_file(video_id, "Instrument")
        action_df = self.read_label_file(video_id, "Action")
        
        # Get sampled frame indices
        sampled_frames = range(0, len(phase_df), sample_interval)
        
        # Prepare data structure
        data = {
            "video_id": video_id,
            "frames": {},
            "total_frames": len(sampled_frames),
            "sample_interval": sample_interval,
            "label_mapping": {
                "phases": self.PHASE_MAPPING,
                "instruments": {i: name for i, name in enumerate(self.INSTRUMENTS)},
                "actions": {i: name for i, name in enumerate(self.ACTIONS)}
            }
        }
        
        # Process each sampled frame
        for frame in sampled_frames:
            try:
                phase_id = int(phase_df.iloc[frame, 1])
                data["frames"][str(frame)] = {
                    "phase": {
                        "id": phase_id,
                        "name": self.PHASE_MAPPING[phase_id]
                    },
                    "instruments": {
                        name: int(instrument_df.iloc[frame, i+1])
                        for i, name in enumerate(self.INSTRUMENTS)
                    },
                    "actions": {
                        name: int(action_df.iloc[frame, i+1])
                        for i, name in enumerate(self.ACTIONS)
                    }
                }
            except Exception as e:
                logging.error(f"Error processing frame {frame} of {video_id}: {str(e)}")
                continue
        
        return data

    def save_labels(self, data: Dict, video_id: str):
        """Save combined labels to JSON file."""
        output_path = self.output_dir / f'{video_id}_combined_labels.json'
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved combined labels to {output_path}")

def main():
    """Main entry point with predefined video groups."""
    
    # Configure videos and their sampling intervals
    VIDEO_GROUPS = {
        50: [  # 50-frame sampling
            'Hei-Chole24', 'Hei-Chole23', 'Hei-Chole20',
            'Hei-Chole19', 'Hei-Chole18', 'Hei-Chole17',
            'Hei-Chole16'
        ],
        25: [  # 25-frame sampling
            f'Hei-Chole{i}' for i in range(1, 25)
            if f'Hei-Chole{i}' not in [
                '16', '17', '18', '19', '20', '23', '24'
            ]
        ]
    }
    
    # Initialize processor
    processor = HeiCholeLabelProcessor(
        base_path='/path/to/heichole/dataset'  # Replace with actual path
    )
    
    # Process all videos
    for interval, videos in VIDEO_GROUPS.items():
        for video_id in videos:
            try:
                data = processor.create_sampled_labels(video_id, interval)
                processor.save_labels(data, video_id)
            except Exception as e:
                logging.error(f"Failed to process {video_id}: {str(e)}")
                continue

if __name__ == "__main__":
    main()