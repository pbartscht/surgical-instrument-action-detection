import os
import shutil
import random
from pathlib import Path

def create_split(base_dir, output_dir, split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}):
    # Create base directories
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    domain_adapt_dir = output_dir / 'domain_adaptation'
    
    # Get list of all video directories
    video_dirs = [d for d in (base_dir / 'Videos').glob('VID*') if d.is_dir()]
    random.shuffle(video_dirs)  # Randomize the order
    
    # Calculate split sizes
    total_videos = len(video_dirs)
    train_size = int(total_videos * split_ratios['train'])
    val_size = int(total_videos * split_ratios['val'])
    test_size = total_videos - train_size - val_size
    
    # Split the video list
    train_videos = video_dirs[:train_size]
    val_videos = video_dirs[train_size:train_size + val_size]
    test_videos = video_dirs[train_size + val_size:]
    
    splits = {
        'train': train_videos,
        'val': val_videos,
        'test': test_videos
    }
    
    # Print split information
    print(f"\nTotal videos: {total_videos}")
    print(f"Training videos: {len(train_videos)}")
    print(f"Validation videos: {len(val_videos)}")
    print(f"Test videos: {len(test_videos)}")
    
    # Create directories and copy files
    for split_name, videos in splits.items():
        # Create directories for this split
        split_dir = domain_adapt_dir / split_name
        video_dir = split_dir / 'Videos'
        label_dir = split_dir / 'Labels'
        
        video_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying {split_name} split:")
        print(f"Videos in {split_name}: {[v.name for v in videos]}")
        
        for video_path in videos:
            video_name = video_path.name
            print(f"Processing {video_name}")
            
            # Copy video directory
            dest_video = video_dir / video_name
            shutil.copytree(video_path, dest_video)
            
            # Copy corresponding label file
            label_file = base_dir / 'Labels' / f"{video_name}.json"
            if label_file.exists():
                shutil.copy2(label_file, label_dir)
            else:
                print(f"Warning: Label file not found for {video_name}")

if __name__ == "__main__":
    base_dir = "/data/Bartscht/HeiChole"
    output_dir = "/data/Bartscht/HeiChole"
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Starting dataset split...")
    print(f"Reading from: {base_dir}")
    print(f"Writing to: {output_dir}/domain_adaptation")
    
    create_split(base_dir, output_dir)
    
    print("\nDataset split complete!")
    print("\nFinal structure:")
    print("/data/Bartscht/HeiChole/domain_adaptation/")
    print("├── train/")
    print("│   ├── Videos/")
    print("│   └── Labels/")
    print("├── val/")
    print("│   ├── Videos/")
    print("│   └── Labels/")
    print("└── test/")
    print("    ├── Videos/")
    print("    └── Labels/")