import json
from pathlib import Path
from collections import defaultdict

class HeiCholeLoader:
    def __init__(self, dataset_path):
        """
        Initialize the HeiChole dataset loader
        
        Args:
            dataset_path (str or Path): Path to the HeiChole dataset root directory
        """
        self.dataset_path = Path(dataset_path)
        self.labels_path = self.dataset_path / "Labels"
        
        # Define all possible labels
        self.instrument_labels = [
            'grasper', 'clipper', 'coagulation', 'scissors',
            'suction_irrigation', 'specimen_bag', 'stapler'
        ]
        
        self.action_labels = [
            'grasp', 'hold', 'cut', 'clip'
        ]

    def load_single_video(self, video_id):
        """
        Load annotations for a single video
        
        Args:
            video_id (str): Video identifier (e.g., "VID01")
            
        Returns:
            dict: Frame-wise binary annotations
        """
        json_path = self.labels_path / f"{video_id}.json"
        
        # Dictionary to store binary annotations per frame
        frame_annotations = defaultdict(lambda: {
            'instruments': {label: 0 for label in self.instrument_labels},
            'actions': {label: 0 for label in self.action_labels}
        })
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                frames = data.get('frames', {})
                
                for frame_num, frame_data in frames.items():
                    frame_number = int(frame_num)
                    
                    # Process instruments (binary)
                    instruments = frame_data.get('instruments', {})
                    for instr_name, present in instruments.items():
                        if instr_name in self.instrument_labels:
                            # Convert to binary: 1 if present (value > 0), 0 if not
                            frame_annotations[frame_number]['instruments'][instr_name] = 1 if present > 0 else 0
                    
                    # Process actions (binary)
                    actions = frame_data.get('actions', {})
                    for action_name, present in actions.items():
                        if action_name in self.action_labels:
                            # Convert to binary: 1 if present (value > 0), 0 if not
                            frame_annotations[frame_number]['actions'][action_name] = 1 if present > 0 else 0
            
            return frame_annotations
            
        except Exception as e:
            print(f"Error loading annotations for {video_id}: {str(e)}")
            return None

    def analyze_dataset(self):
        """
        Analyze the entire dataset and compute label distributions
        
        Returns:
            tuple: (statistics, total_frames)
            - statistics: dict with label frequencies
            - total_frames: total number of frames analyzed
        """
        # Get all video IDs
        video_ids = [f.stem for f in self.labels_path.glob("*.json")]
        
        # Initialize counters
        stats = {
            'instruments': defaultdict(int),
            'actions': defaultdict(int)
        }
        total_frames = 0
        
        print(f"\nAnalyzing {len(video_ids)} videos...")
        
        # Process each video
        for video_id in video_ids:
            annotations = self.load_single_video(video_id)
            if annotations:
                for frame_data in annotations.values():
                    total_frames += 1
                    
                    # Count instrument occurrences
                    for instr, present in frame_data['instruments'].items():
                        if present > 0:
                            stats['instruments'][instr] += 1
                    
                    # Count action occurrences
                    for action, present in frame_data['actions'].items():
                        if present > 0:
                            stats['actions'][action] += 1
        
        return stats, total_frames

    def print_analysis(self, stats, total_frames):
        """
        Print a formatted analysis of the dataset
        
        Args:
            stats (dict): Statistics from analyze_dataset
            total_frames (int): Total number of frames analyzed
        """
        print("\n====== HEICHOLE DATASET ANALYSIS ======")
        print(f"Total frames analyzed: {total_frames}")
        
        # Print instrument statistics
        print("\nINSTRUMENTS:")
        print(f"{'Label':20s} {'Count':>8s} {'% of Frames':>12s}")
        print("-" * 42)
        
        for instr in self.instrument_labels:
            count = stats['instruments'][instr]
            percentage = (count / total_frames) * 100
            print(f"{instr:20s} {count:8d} {percentage:11.2f}%")
        
        # Print action statistics
        print("\nACTIONS:")
        print(f"{'Label':20s} {'Count':>8s} {'% of Frames':>12s}")
        print("-" * 42)
        
        for action in self.action_labels:
            count = stats['actions'][action]
            percentage = (count / total_frames) * 100
            print(f"{action:20s} {count:8d} {percentage:11.2f}%")

def main():
    # Pfad zum Dataset anpassen
    dataset_path = "/data/Bartscht/HeiChole/domain_adaptation/test"
    
    # Initialize loader
    loader = HeiCholeLoader(dataset_path)
    
    # Analyze dataset
    stats, total_frames = loader.analyze_dataset()
    
    # Print results
    loader.print_analysis(stats, total_frames)

if __name__ == "__main__":
    main()