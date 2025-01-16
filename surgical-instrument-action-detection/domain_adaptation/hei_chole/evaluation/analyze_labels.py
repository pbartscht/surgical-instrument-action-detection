import os
from pathlib import Path
import json
from collections import defaultdict

def analyze_label_structure(dataset_dir):
    """
    Analyzes the structure of labels across all JSON files and identifies differences.
    """
    labels_dir = Path(dataset_dir) / "Labels"
    
    # Store structure information per video
    structures = {}
    all_instruments = set()
    all_actions = set()
    
    print("\nAnalyzing label structure across all videos...")
    
    # First pass: collect all unique labels and basic structure
    for json_file in sorted(labels_dir.glob("*.json")):
        video_id = json_file.stem
        structures[video_id] = {
            'instruments': set(),
            'actions': set(),
            'frame_count': 0,
            'sample_frame': None,  # Store one complete frame for structure analysis
            'value_types': defaultdict(set)  # Store what types of values we see
        }
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                frames = data.get('frames', {})
                structures[video_id]['frame_count'] = len(frames)
                
                # Analyze each frame
                for frame_num, frame_data in frames.items():
                    # Store first frame as sample
                    if structures[video_id]['sample_frame'] is None:
                        structures[video_id]['sample_frame'] = frame_data
                    
                    # Collect instruments
                    instruments = frame_data.get('instruments', {})
                    structures[video_id]['instruments'].update(instruments.keys())
                    all_instruments.update(instruments.keys())
                    
                    # Store value types for instruments
                    for instr, value in instruments.items():
                        structures[video_id]['value_types'][f'instruments.{instr}'].add(
                            str(type(value).__name__)
                        )
                    
                    # Collect actions
                    actions = frame_data.get('actions', {})
                    structures[video_id]['actions'].update(actions.keys())
                    all_actions.update(actions.keys())
                    
                    # Store value types for actions
                    for action, value in actions.items():
                        structures[video_id]['value_types'][f'actions.{action}'].add(
                            str(type(value).__name__)
                        )
                    
        except Exception as e:
            print(f"Error processing {video_id}: {str(e)}")
            continue
    
    # Print analysis results
    print("\n=== LABEL STRUCTURE ANALYSIS ===")
    print(f"\nTotal number of videos analyzed: {len(structures)}")
    
    # Check for structural differences
    print("\n=== INSTRUMENT ANALYSIS ===")
    print(f"\nTotal unique instruments across all videos: {len(all_instruments)}")
    print("\nInstruments and their presence in videos:")
    print("=" * 80)
    print(f"{'Instrument':20s} {'Present in Videos':>15s} {'Value Types':>20s}")
    print("-" * 80)
    
    for instrument in sorted(all_instruments):
        videos_with_instrument = sum(1 for vid_struct in structures.values() 
                                   if instrument in vid_struct['instruments'])
        # Collect all value types for this instrument
        value_types = set()
        for vid_struct in structures.values():
            value_types.update(vid_struct['value_types'][f'instruments.{instrument}'])
        
        print(f"{instrument:20s} {videos_with_instrument:15d} {', '.join(value_types):>20s}")
    
    # Check for videos missing instruments
    print("\nVideos missing instruments:")
    for video_id, struct in sorted(structures.items()):
        missing = all_instruments - struct['instruments']
        if missing:
            print(f"{video_id}: Missing {', '.join(sorted(missing))}")
    
    print("\n=== ACTION ANALYSIS ===")
    print(f"\nTotal unique actions across all videos: {len(all_actions)}")
    print("\nActions and their presence in videos:")
    print("=" * 80)
    print(f"{'Action':20s} {'Present in Videos':>15s} {'Value Types':>20s}")
    print("-" * 80)
    
    for action in sorted(all_actions):
        videos_with_action = sum(1 for vid_struct in structures.values() 
                               if action in vid_struct['actions'])
        # Collect all value types for this action
        value_types = set()
        for vid_struct in structures.values():
            value_types.update(vid_struct['value_types'][f'actions.{action}'])
        
        print(f"{action:20s} {videos_with_action:15d} {', '.join(value_types):>20s}")
    
    # Check for videos missing actions
    print("\nVideos missing actions:")
    for video_id, struct in sorted(structures.items()):
        missing = all_actions - struct['actions']
        if missing:
            print(f"{video_id}: Missing {', '.join(sorted(missing))}")
    
    # Sample structure
    print("\n=== SAMPLE FRAME STRUCTURE ===")
    if structures:
        sample_video = next(iter(structures.values()))
        if sample_video['sample_frame']:
            print("\nExample frame structure:")
            print(json.dumps(sample_video['sample_frame'], indent=2))

if __name__ == "__main__":
    dataset_dir = Path("/data/Bartscht/HeiChole")
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
    else:
        analyze_label_structure(dataset_dir)