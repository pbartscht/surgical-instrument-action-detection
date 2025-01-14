import json
import os
import shutil
from pathlib import Path

def create_video_folders(input_base_30fps, json_dir_1fps, output_base_1fps):
    """
    Creates folder structure and copies relevant frames based on the structured JSONs.
    This script processes the restructured JSON files and extracts corresponding frames
    from the 30fps dataset to create the 1fps version.
    
    Args:
        input_base_30fps (str): Path to the original 30fps data directory
        json_dir_1fps (str): Directory containing the restructured JSON files
        output_base_1fps (str): Directory where the 1fps frames will be saved
    """
    # Find all structured JSON files
    json_files = [
        f for f in os.listdir(json_dir_1fps) 
        if f.endswith('_structured.json')
    ]
    print(f"Found {len(json_files)} JSON files in {json_dir_1fps}")
    
    # Process each structured JSON file
    for json_file in sorted(json_files):  # Sort for consistent processing order
        # Load JSON data
        json_path = os.path.join(json_dir_1fps, json_file)
        print(f"\nProcessing JSON: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract video information from JSON
        video_id = data['video_id']          # e.g., VID01
        original_case = data['original_case'] # e.g., CASE01
        
        # Create destination directory for the current video
        video_dir = os.path.join(output_base_1fps, video_id)
        os.makedirs(video_dir, exist_ok=True)
        
        # Define source directory for original frames
        source_dir = os.path.join(input_base_30fps, original_case)
        
        print(f"Source directory: {source_dir}")
        print(f"Target directory: {video_dir}")
        
        # Copy selected frames from 30fps to 1fps
        frames_copied = 0
        for frame_name in sorted(data['frames'].keys()):  # Sort for consistent processing
            # Construct source and destination paths
            src_path = os.path.join(source_dir, frame_name)
            dst_path = os.path.join(video_dir, frame_name)
            
            # Copy frame if it exists in source
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)  # copy2 preserves metadata
                frames_copied += 1
            else:
                print(f"Warning: Source image not found: {src_path}")
        
        print(f"Completed {video_id}: {frames_copied} frames copied")

def process_split(input_base, output_base, split="train"):
    """
    Process either training or test split of the dataset.
    
    Args:
        input_base (str): Base directory containing original 30fps data
        output_base (str): Base directory for saving processed 1fps data
        split (str): Dataset split to process ("train" or "test")
    """
    # Construct paths for the specified split
    input_base_30fps = os.path.join(input_base, split)
    json_dir_1fps = os.path.join(output_base, split, "structured_jsons")
    output_base_1fps = os.path.join(output_base, split)
    
    # Validate directories
    if not os.path.exists(input_base_30fps):
        raise FileNotFoundError(f"Input directory not found: {input_base_30fps}")
    if not os.path.exists(json_dir_1fps):
        raise FileNotFoundError(f"JSON directory not found: {json_dir_1fps}")
    
    # Create folder structure and copy images
    create_video_folders(input_base_30fps, json_dir_1fps, output_base_1fps)
    print(f"\nFolder structure creation complete for {split} split!")

if __name__ == "__main__":
    # Define base directories
    input_base = "/path/to/GrasP/30fps"  #replace with your path
    output_base = "/path/to/GrasP/1fps"  #replace with your path
    
    # Process training split
    print("Processing training split...")
    process_split(input_base, output_base, "train")
    
    # Process test split
    print("\nProcessing test split...")
    process_split(input_base, output_base, "test")