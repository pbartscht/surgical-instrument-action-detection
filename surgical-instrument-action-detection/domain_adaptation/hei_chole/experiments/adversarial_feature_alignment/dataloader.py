import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random
from pathlib import Path
from label_loader import SurgicalDataset, TOOL_MAPPING, HEICHOLE_CLASSES
import logging


class BalancedBatchSampler(Sampler):
    """Sampler that ensures equal representation of both domains in each batch"""
    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size
        self.source_indices = list(range(dataset.source_len))
        self.target_indices = list(range(dataset.source_len, len(dataset)))
        self.num_batches = min(len(self.source_indices), len(self.target_indices)) // (batch_size // 2)

    def __iter__(self):
        # Listen mischen
        random.shuffle(self.source_indices)
        random.shuffle(self.target_indices)
        
        for i in range(self.num_batches):
            # 16 source samples nehmen
            start_src = i * (self.batch_size // 2)
            end_src = (i + 1) * (self.batch_size // 2)
            source_batch = self.source_indices[start_src:end_src]
            
            # 16 target samples nehmen
            start_tgt = i * (self.batch_size // 2)
            end_tgt = (i + 1) * (self.batch_size // 2)
            target_batch = self.target_indices[start_tgt:end_tgt]
            
            # Beide zusammenfügen und mischen
            batch = source_batch + target_batch
            random.shuffle(batch)
            
            # Gesamten Batch auf einmal zurückgeben
            yield batch

    def __len__(self):
        return self.num_batches

class CombinedDataset(Dataset):
    def __init__(self, source_dir, target_dir):
        self.source_dataset = SurgicalDataset(source_dir, 'source')
        self.target_dataset = SurgicalDataset(target_dir, 'target')
        
        #self.mapping_matrix = self.create_mapping_matrix()
        self.source_len = len(self.source_dataset)
        self.target_len = len(self.target_dataset)
        
        print(f"\nDataset Statistiken:")
        print(f"CholecT50 Samples: {self.source_len}")
        print(f"HeiChole Samples: {self.target_len}")
        
    
    def __len__(self):
        return self.source_len + self.target_len
    
    def __getitem__(self, idx):
        base_sample = {
            'image': None,
            'labels': None,
            'domain': None,
            'video': None,
            'frame': None
        }
        
        if idx < self.source_len:
            sample = self.source_dataset[idx]
            base_sample.update({
                'image': sample['image'],
                'labels': sample['labels'],
                'domain': torch.tensor(0),
                'video': sample['video'],
                'frame': sample['frame']
            })
        else:
            sample = self.target_dataset[idx - self.source_len]
            base_sample.update({
                'image': sample['image'],
                'labels': sample['labels'][:6],
                'domain': torch.tensor(1),
                'video': sample['video'],
                'frame': sample['frame']
            })
        
        return base_sample

def balanced_dataloader(split='train', batch_size=32):
    """
    Erstellt einen balancierten DataLoader für Training, Validation oder Test
    
    Args:
        split (str): 'train', 'val' oder 'test'
        batch_size (int): Batch Size
    """
    CONFIG = {
        'source_dir': Path("/data/Bartscht/CholecT50"),
        'target_dir': Path(f"/data/Bartscht/HeiChole/domain_adaptation/{split}"),
        'batch_size': batch_size
    }
    
    combined_dataset = CombinedDataset(CONFIG['source_dir'], CONFIG['target_dir'])
    balanced_sampler = BalancedBatchSampler(combined_dataset, CONFIG['batch_size'])
    
    dataloader = DataLoader(
        combined_dataset,
        batch_sampler=balanced_sampler,
        num_workers=4
    )
    
    # Debug Info nur für Training ausgeben
    if split == 'train':
        for batch in dataloader:
            print("\nErster Batch Shapes:")
            for key, value in batch.items():
                if torch.is_tensor(value):
                    print(f"{key}: {value.shape}")
                else:
                    print(f"{key}: {type(value)} mit Länge {len(value)}")
            break
    
    return dataloader

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataloader = balanced_dataloader()