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

HEICHOLE_CLASSES = {
    0: 'grasper',
    1: 'clipper',
    2: 'coagulation',
    3: 'scissors',
    4: 'suction_irrigation',
    5: 'specimen_bag',
    6: 'stapler'
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
        """Lade und verarbeite die Datensätze basierend auf dem Typ"""
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
        
        if not videos_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError(f"Directory not found: {videos_dir} or {labels_dir}")

        video_count = 0
        for video_dir in videos_dir.glob("*"):
            if not video_dir.is_dir():
                continue
                
            video_count += 1
            video_name = video_dir.name
            logging.info(f"Processing video: {video_name}")
            
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
                self._process_cholect_annotations(video_dir, video_name, annotations)
            else:
                # Process HeiChole format
                self._process_heichole_annotations(video_dir, video_name, annotations)
            
            logging.info(f"Processed {frame_count} frames from video {video_name}")
        
        logging.info(f"Processed {video_count} videos in total")
    
    def _process_cholect_annotations(self, video_dir, video_name, annotations):
        """Verarbeite CholecT50 Annotationen"""
        frame_annotations = defaultdict(lambda: defaultdict(int))
        
        # Extract frame annotations
        for frame, instances in annotations['annotations'].items():
            frame_number = int(frame)
            for instance in instances:
                instrument_id = instance[1]  # Instrument ID is at index 1
                if isinstance(instrument_id, int) and 0 <= instrument_id < 6:
                    instrument_name = TOOL_MAPPING[instrument_id]
                    frame_annotations[frame_number][instrument_name] = 1
        
        # Process frames
        for frame_file in video_dir.glob("*.png"):
            frame_number = int(frame_file.stem)
            
            if frame_number in frame_annotations:
                labels = torch.zeros(len(TOOL_MAPPING))
                for idx, tool_name in TOOL_MAPPING.items():
                    if frame_annotations[frame_number].get(tool_name, 0) > 0:
                        labels[idx] = 1
                
                self.validate_labels(labels, 'source')
                
                self.samples.append({
                    'image_path': str(frame_file),
                    'labels': labels,
                    'video': video_name,
                    'frame': frame_number,
                    'domain': 0
                })
    
    def _process_heichole_annotations(self, video_dir, video_name, annotations):
        """Verarbeite HeiChole Annotationen"""
        for frame_file in video_dir.glob("*.png"):
            frame_number = int(frame_file.stem)
            frame_data = annotations['frames'].get(str(frame_number), {})
            instruments = frame_data.get('instruments', {})
            
            labels = torch.zeros(len(HEICHOLE_CLASSES))
            for instr_name, present in instruments.items():
                if present > 0:
                    for idx, name in HEICHOLE_CLASSES.items():
                        if name == instr_name:
                            labels[idx] = 1
                            break
            
            self.validate_labels(labels, 'target')
            
            self.samples.append({
                'image_path': str(frame_file),
                'labels': labels,
                'video': video_name,
                'frame': frame_number,
                'domain': 1
            })
    
    def validate_labels(self, labels, dataset_type):
        """Validiere Label Format und Dimensionen"""
        if dataset_type == 'source':
            assert labels.size(0) == len(TOOL_MAPPING), \
                f"CholecT50 label should have {len(TOOL_MAPPING)} classes"
        else:
            assert labels.size(0) == len(HEICHOLE_CLASSES), \
                f"HeiChole label should have {len(HEICHOLE_CLASSES)} classes"
        
        # Check wenn Binary
        assert torch.all((labels == 0) | (labels == 1)), \
            "Labels should be binary (0 or 1)"
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Lade und verarbeite ein einzelnes Sample"""
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
    
    def get_statistics(self):
        """Berechne Statistiken über den Datensatz"""
        stats = {
            'total_samples': len(self.samples),
            'videos': set(),
            'class_distribution': torch.zeros(len(HEICHOLE_CLASSES) if self.dataset_type == 'target' else len(TOOL_MAPPING))
        }
        
        for sample in self.samples:
            stats['videos'].add(sample['video'])
            stats['class_distribution'] += sample['labels']
        
        stats['videos'] = len(stats['videos'])
        stats['class_distribution'] = stats['class_distribution'].tolist()
        
        return stats