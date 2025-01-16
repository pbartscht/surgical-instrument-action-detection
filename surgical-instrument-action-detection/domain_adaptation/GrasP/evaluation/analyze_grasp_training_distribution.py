from pathlib import Path
import json
from collections import defaultdict

def analyze_grasp_distribution(dataset_path):
    """
    Analyzes the distribution of instruments and actions in GraSP training dataset.
    
    Args:
        dataset_path: Path to the GraSP dataset directory
    """
    labels_dir = Path(dataset_path) / "Labels"
    video_ids = [f.stem for f in labels_dir.glob("*.json")]
    video_ids.sort()

    # Initialize counters
    total_frames = 0
    instrument_counts = defaultdict(int)
    action_counts = defaultdict(int)
    instrument_action_pairs = defaultdict(lambda: defaultdict(int))

    # Process each video
    for video_id in video_ids:
        json_file = labels_dir / f"{video_id}.json"
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Get categories lookup
                instrument_categories = {
                    cat['id']: cat['name'] 
                    for cat in data['categories']['instruments']
                }
                action_categories = {
                    cat['id']: cat['name'] 
                    for cat in data['categories']['actions']
                }
                
                frames = data.get('frames', {})
                total_frames += len(frames)
                
                # Process each frame
                for frame_data in frames.values():
                    for instrument_ann in frame_data.get('instruments', []):
                        category_id = instrument_ann.get('category_id')
                        if category_id is not None:
                            instr_name = instrument_categories[category_id]
                            instrument_counts[instr_name] += 1
                            
                            # Process actions for this instrument
                            for action_id in instrument_ann.get('actions', []):
                                action_name = action_categories[action_id]
                                action_counts[action_name] += 1
                                instrument_action_pairs[instr_name][action_name] += 1
        
        except Exception as e:
            print(f"Error processing {video_id}: {str(e)}")

    # Print results
    print(f"\nTotal Frames Analyzed: {total_frames}")
    
    # Print instrument frequencies
    print("\nINSTRUMENT FREQUENCIES:")
    for instr_name, count in sorted(instrument_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_frames) * 100
        print(f"{instr_name:30s} {count:8d} {percentage:11.2f}%")
        
        # Print associated actions
        if instrument_action_pairs[instr_name]:
            print("\n  Associated Actions:")
            for action, action_count in sorted(instrument_action_pairs[instr_name].items(), key=lambda x: x[1], reverse=True):
                action_percentage = (action_count / count) * 100
                print(f"    {action:26s}: {action_count:6d} ({action_percentage:5.2f}%)")
            print()

    # Print overall action frequencies
    print("\nOVERALL ACTION FREQUENCIES:")
    for action_name, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_frames) * 100
        print(f"{action_name:30s} {count:8d} {percentage:11.2f}%")

if __name__ == '__main__':
    DATASET_PATH = "/data/Bartscht/GrasP/train"  # Changed to train folder
    analyze_grasp_distribution(DATASET_PATH)