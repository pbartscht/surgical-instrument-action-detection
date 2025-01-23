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
        
        self.mapping_matrix = self.create_mapping_matrix()
        self.source_len = len(self.source_dataset)
        self.target_len = len(self.target_dataset)
        
        print(f"\nDataset Statistiken:")
        print(f"CholecT50 Samples: {self.source_len}")
        print(f"HeiChole Samples: {self.target_len}")
        
    def create_mapping_matrix(self):
        mapping_matrix = torch.zeros(6, 5)
        mapping_matrix[0, 0] = 1  # grasper -> grasper
        mapping_matrix[1, 2] = 1  # bipolar -> coagulation
        mapping_matrix[2, 2] = 1  # hook -> coagulation
        mapping_matrix[3, 3] = 1  # scissors -> scissors
        mapping_matrix[4, 1] = 1  # clipper -> clipper
        mapping_matrix[5, 4] = 1  # irrigator -> suction_irrigation
        return mapping_matrix
    
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
            mapped_labels = torch.matmul(sample['labels'].float(), self.mapping_matrix)
            base_sample.update({
                'image': sample['image'],
                'labels': mapped_labels,
                'domain': torch.tensor(0),
                'video': sample['video'],
                'frame': sample['frame']
            })
        else:
            sample = self.target_dataset[idx - self.source_len]
            base_sample.update({
                'image': sample['image'],
                'labels': sample['labels'][:5],
                'domain': torch.tensor(1),
                'video': sample['video'],
                'frame': sample['frame']
            })
        
        return base_sample

def test_dataloader():
    CONFIG = {
        'source_dir': Path("/data/Bartscht/CholecT50"),
        'target_dir': Path("/data/Bartscht/HeiChole/domain_adaptation/train"),
        'batch_size': 32
    }
    
    combined_dataset = CombinedDataset(CONFIG['source_dir'], CONFIG['target_dir'])
    
    # Balanced Sampler verwenden
    balanced_sampler = BalancedBatchSampler(combined_dataset, CONFIG['batch_size'])
    
    dataloader = DataLoader(
        combined_dataset,
        batch_sampler=balanced_sampler,
        num_workers=4
    )
    
    # Test mehrere Batches
    print("\nTeste Batch-Balance über mehrere Batches:")
    total_source = 0
    total_target = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 5:  # Teste erste 5 Batches
            break
            
        domain_counts = torch.bincount(batch['domain'])
        total_source += domain_counts[0].item()
        total_target += domain_counts[1].item()
        
        print(f"\nBatch {batch_idx}:")
        print(f"Source (CholecT50): {domain_counts[0].item()} samples")
        print(f"Target (HeiChole): {domain_counts[1].item()} samples")
    
    print(f"\nGesamtverteilung über {batch_idx + 1} Batches:")
    print(f"Source (CholecT50): {total_source//(batch_idx + 1)} samples pro Batch")
    print(f"Target (HeiChole): {total_target//(batch_idx + 1)} samples pro Batch")
    
    return dataloader

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataloader = test_dataloader()