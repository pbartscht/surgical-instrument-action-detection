import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
import torchvision.transforms as transforms
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING = {
    'grasper': 'grasper',
    'bipolar': 'coagulation',
    'clipper': 'clipper',
    'hook': 'coagulation',
    'scissors': 'scissors',
    'irrigator': 'suction_irrigation'
}

TOOL_MAPPING = {
    0: 'grasper', 
    1: 'bipolar', 
    2: 'hook',
    3: 'scissors', 
    4: 'clipper', 
    5: 'irrigator'
}



class SurgicalDataset(Dataset):
    def __init__(self, dataset_dir, dataset_type='source', transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_type = dataset_type
        self.transform = transform or transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        logging.info(f"Initializing dataset with dir: {self.dataset_dir}, type: {dataset_type}")
        
        self.samples = []
        self._load_dataset()
        logging.info(f"Loaded {len(self.samples)} samples")
        
    def _load_dataset(self):
        if self.dataset_type == 'source':
            # CholecT50 structure
            videos_dir = self.dataset_dir / "videos"
            labels_dir = self.dataset_dir / "labels"
        else:
            # HeiChole structure
            videos_dir = self.dataset_dir / "Videos"
            labels_dir = self.dataset_dir / "Labels"
            
        logging.info(f"Loading from videos_dir: {videos_dir}")
        logging.info(f"Loading from labels_dir: {labels_dir}")
        
        if not videos_dir.exists():
            logging.error(f"Videos directory does not exist: {videos_dir}")
            return
            
        if not labels_dir.exists():
            logging.error(f"Labels directory does not exist: {labels_dir}")
            return

        video_count = 0
        for video_dir in videos_dir.glob("*"):
            if not video_dir.is_dir():
                continue
                
            video_count += 1
            video_name = video_dir.name
            logging.info(f"Processing video: {video_name}")
            
            if self.dataset_type == 'source':
                # Load CholecT50 labels
                json_file = labels_dir / f"{video_name}.json"
            else:
                # Load HeiChole labels
                json_file = labels_dir / f"{video_name}.json"
            
            if not json_file.exists():
                logging.warning(f"Label file does not exist: {json_file}")
                continue
                
            logging.info(f"Loading annotations from: {json_file}")
            
            try:
                with open(json_file, 'r') as f:
                    annotations = json.load(f)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON file {json_file}: {e}")
                continue
            
            frame_count = 0
            if self.dataset_type == 'source':
                # Process CholecT50 format
                frame_annotations = defaultdict(lambda: defaultdict(int))
                
                # Extract frame annotations
                for frame, instances in annotations['annotations'].items():
                    frame_number = int(frame)
                    for instance in instances:
                        instrument_id = instance[1]  # Instrument ID is at index 1
                        if isinstance(instrument_id, int) and 0 <= instrument_id < 6:
                            instrument_name = TOOL_MAPPING[instrument_id]
                            frame_annotations[frame_number][instrument_name] = 1
                
                # Process frames - look for .png files with 6-digit frame numbers
                for frame_file in video_dir.glob("*.png"):
                    # Frame name format: XXXXXX.png (e.g., 001720.png)
                    frame_number = int(frame_file.stem)  # This will convert "001720" to 1720
                    
                    logging.debug(f"Processing frame {frame_number} from {frame_file}")
                    
                    if frame_number in frame_annotations:
                        labels = torch.zeros(len(TOOL_MAPPING))
                        for idx, tool_name in TOOL_MAPPING.items():
                            if frame_annotations[frame_number].get(tool_name, 0) > 0:
                                labels[idx] = 1
                        
                        frame_count += 1
                        self.samples.append({
                            'image_path': str(frame_file),
                            'labels': labels,
                            'video': video_name,
                            'frame': frame_number,
                            'domain': 0
                        })
            else:
                # Process HeiChole format (unchanged)
                for frame_file in video_dir.glob("*.png"):
                    frame_number = int(frame_file.stem)
                    frame_data = annotations['frames'].get(str(frame_number), {})
                    instruments = frame_data.get('instruments', {})
                    
                    labels = torch.zeros(len(TOOL_MAPPING))
                    for instr_name, present in instruments.items():
                        if present > 0:
                            for idx, cholect_instr in TOOL_MAPPING.items():
                                mapped_name = CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING.get(cholect_instr)
                                if mapped_name == instr_name:
                                    labels[idx] = 1
                                    break
                    
                    frame_count += 1
                    self.samples.append({
                        'image_path': str(frame_file),
                        'labels': labels,
                        'video': video_name,
                        'frame': frame_number,
                        'domain': 1
                    })
                
            logging.info(f"Processed {frame_count} frames from video {video_name}")
            
        logging.info(f"Processed {video_count} videos in total")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample['image_path'])
        except Exception as e:
            logging.error(f"Error loading image {sample['image_path']}: {e}")
            raise
            
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'labels': sample['labels'],
            'video': sample['video'],
            'frame': sample['frame'],
            'domain': sample['domain']
        }