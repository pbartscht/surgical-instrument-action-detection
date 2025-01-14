import json
import os
from pathlib import Path

def reorganize_grasp_json(input_json_path, case_name, output_dir):
    """
    Reorganizes the GraSP JSON annotations into a more hierarchical structure.
    
    Args:
        input_json_path (str): Path to the original JSON annotation file
        case_name (str): Name of the case/video being processed (e.g., 'CASE01')
        output_dir (str): Directory where the restructured JSON will be saved
    
    Returns:
        str: Path to the generated structured JSON file
    """
    # Read input JSON file
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Initialize new hierarchical structure
    new_data = {
        "video_id": f"VID{case_name[-2:]}",  # Extract video ID from case name (e.g., CASE01 -> VID01)
        "original_case": case_name,
        "metadata": {
            "width": 1280,
            "height": 800,
            "fps": 1  # 1fps after sampling from original 30fps
        },
        "categories": {
            "instruments": data.get("categories", []),       # Surgical instruments
            "actions": data.get("actions_categories", []),   # Surgical actions/verbs
            "phases": data["phases_categories"],            # Surgical phases
            "steps": data["steps_categories"]               # Surgical steps
        },
        "frames": {}
    }
    
    # Create dictionary for organizing annotations by frame
    frame_annotations = {}
    
    # Create lookup dictionary for images using their IDs
    # Filter only images belonging to the current case
    images_dict = {
        img["id"]: img for img in data["images"]
        if img["file_name"].startswith(f"{case_name}/")
    }
    
    # Process annotations and organize them by frame
    for ann in data["annotations"]:
        if ann["image_id"] in images_dict:
            image = images_dict[ann["image_id"]]
            frame_num = image["frame_num"]
            frame_name = os.path.basename(image["file_name"])
            
            # Initialize frame data structure if not exists
            if frame_name not in frame_annotations:
                frame_annotations[frame_name] = {
                    "frame_num": frame_num,
                    "file_name": frame_name,
                    "phase": ann["phases"],
                    "step": ann["steps"],
                    "instruments": []
                }
            
            # Compile instrument information including location and actions
            instrument_info = {
                "id": ann["id"],
                "category_id": ann["category_id"],
                "area": ann.get("area", 0),                    # Object area in pixels
                "bbox": ann.get("bbox", []),                   # Bounding box coordinates
                "segmentation": ann.get("segmentation", {}),   # Segmentation mask
                "actions": ann.get("actions", []),             # Associated actions
                "iscrowd": ann.get("iscrowd", 0)              # Crowd annotation flag
            }
            frame_annotations[frame_name]["instruments"].append(instrument_info)
    
    # Add organized frames to new structure
    new_data["frames"] = frame_annotations
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output path and save restructured JSON
    output_path = os.path.join(output_dir, f"{new_data['video_id']}_structured.json")
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    return output_path

def process_all_cases(input_base, output_base):
    """
    Process all surgical cases in the input directory.
    
    Args:
        input_base (str): Base directory containing original 30fps data
        output_base (str): Base directory for saving processed 1fps data
    """
    # Path to the main annotation file
    json_path = os.path.join(input_base, "grasp_short-term_train.json") #replace here in second run with grasp_short-term_test.json
    
    # Find all case directories (format: CASE01, CASE02, etc.)
    case_dirs = [
        d for d in os.listdir(input_base)
        if d.startswith("CASE") and os.path.isdir(os.path.join(input_base, d))
    ]
    
    # Process each case
    for case_name in case_dirs:
        print(f"Processing {case_name}...")
        output_dir = os.path.join(output_base, "structured_jsons")
        output_path = reorganize_grasp_json(json_path, case_name, output_dir)
        print(f"Saved restructured JSON to: {output_path}")

if __name__ == "__main__":
    # Define base directories
    input_base = "/path/to/GrasP/30fps/train" #after successfull run replace with test
    output_base = "/path/to/GrasP/1fps/train" #after successfull run replace with test
    
    # Process all cases
    process_all_cases(input_base, output_base)