#!/usr/bin/env python3
"""
HeiChole Dataset Processing Script
================================

This script processes videos from the HeiChole dataset by:
1. Extracting frames at a specified sampling rate
2. Resizing frames to a target resolution
3. Saving frames in a memory-efficient format
4. Providing progress tracking and error handling

Requirements:
- Python 3.7+
- OpenCV (cv2)
- tqdm
"""

import argparse
import cv2
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm

class HeiCholeProcessor:
    """Processes HeiChole dataset videos with configurable parameters."""
    
    def __init__(
        self,
        target_fps: float = 1.0,
        target_size: Tuple[int, int] = (512, 512),
        output_format: str = 'png'
    ):
        """
        Initialize the processor with given parameters.
        
        Args:
            target_fps: Target frames per second for extraction
            target_size: Target frame size as (width, height)
            output_format: Output image format ('png' or 'jpg')
        """
        self.target_fps = target_fps
        self.target_size = target_size
        self.output_format = output_format
        
        if output_format not in ['png', 'jpg']:
            raise ValueError("Output format must be 'png' or 'jpg'")

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        video_id: Optional[str] = None
    ) -> dict:
        """
        Process a single video file.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory for output frames
            video_id: Optional video identifier for output naming
            
        Returns:
            dict: Processing statistics
        """
        # Input validation
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
            
        # Get video properties
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(source_fps / self.target_fps)
        
        # Initialize statistics
        stats = {
            'source_fps': source_fps,
            'target_fps': self.target_fps,
            'total_frames': total_frames,
            'processed_frames': 0,
            'skipped_frames': 0
        }
        
        # Process frames with progress bar
        with tqdm(total=total_frames//frame_interval, 
                 desc=f"Processing {os.path.basename(video_path)}") as pbar:
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    try:
                        # Resize frame
                        resized_frame = cv2.resize(frame, self.target_size)
                        
                        # Generate output path
                        frame_name = f"{frame_count:06d}.{self.output_format}"
                        if video_id:
                            frame_name = f"{video_id}_{frame_name}"
                        frame_path = os.path.join(output_dir, frame_name)
                        
                        # Save frame
                        cv2.imwrite(frame_path, resized_frame)
                        stats['processed_frames'] += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"Error processing frame {frame_count}: {str(e)}")
                        stats['skipped_frames'] += 1
                        
                frame_count += 1
                
        cap.release()
        return stats

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Process HeiChole dataset videos for frame extraction'
    )
    parser.add_argument(
        'video_dir',
        help='Directory containing input videos'
    )
    parser.add_argument(
        'output_dir',
        help='Directory for output frames'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=1.0,
        help='Target frames per second (default: 1.0)'
    )
    parser.add_argument(
        '--size',
        type=int,
        nargs=2,
        default=[512, 512],
        help='Target frame size as width height (default: 512 512)'
    )
    parser.add_argument(
        '--format',
        choices=['png', 'jpg'],
        default='png',
        help='Output image format (default: png)'
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = HeiCholeProcessor(
        target_fps=args.fps,
        target_size=tuple(args.size),
        output_format=args.format
    )
    
    # Process all videos in directory
    video_files = [f for f in os.listdir(args.video_dir) 
                  if f.endswith(('.mp4', '.avi'))]
    
    for video_file in video_files:
        video_path = os.path.join(args.video_dir, video_file)
        video_id = os.path.splitext(video_file)[0]
        output_subdir = os.path.join(args.output_dir, f"VID_{video_id}")
        
        try:
            stats = processor.process_video(
                video_path,
                output_subdir,
                video_id
            )
            print(f"\nProcessing statistics for {video_file}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
        except Exception as e:
            print(f"\nError processing {video_file}: {str(e)}")

if __name__ == '__main__':
    main()