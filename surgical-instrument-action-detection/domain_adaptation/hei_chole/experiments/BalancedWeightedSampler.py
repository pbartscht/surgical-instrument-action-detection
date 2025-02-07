import torch
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
import random
from pathlib import Path
from label_loader import SurgicalDataset, TOOL_MAPPING, HEICHOLE_CLASSES
import logging
import json
import os

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=32, cache_dir="sampler_cache"):
        print("\nInitialisiere CachedDomainBalancedSampler...")
        self.dataset = dataset
        self.batch_size = batch_size
        self.source_len = dataset.source_len
        self.target_indices = list(range(dataset.source_len, len(dataset)))
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "class_indices_cache.json"
        
        # Vorberechnung der Klassengewichte (diese sind fix)
        self.class_counts = torch.tensor([48182, 4402, 36967, 1376, 2466, 3571], dtype=torch.float)
        self.class_weights = 1.0 / (self.class_counts + 1e-6)
        self.class_weights = self.class_weights / self.class_weights.sum()
        
        print("\nKlassengewichte für Sampling:")
        for i, weight in enumerate(self.class_weights):
            print(f"Klasse {i}: {weight:.4f}")
        
        # Lade oder erstelle Cache
        self.source_indices_by_class = self._load_or_create_cache()
        
        # Berechne Anzahl Batches
        self.num_batches = min(self.source_len, len(self.target_indices)) // (batch_size // 2)
        print(f"\nAnzahl Batches pro Epoch: {self.num_batches}")

    def _load_or_create_cache(self):
        """Lädt Cache wenn vorhanden, erstellt ihn sonst"""
        if not self.cache_dir.exists():
            os.makedirs(self.cache_dir)
            
        if self.cache_file.exists():
            print("\nLade vorsortierten Cache...")
            with open(self.cache_file, 'r') as f:
                return [list(map(int, indices)) for indices in json.load(f)]
        
        print("\nErstelle neuen Cache...")
        indices_by_class = [[] for _ in range(len(self.class_weights))]
        
        # Schnellere Vorsortierung durch Batch-Verarbeitung
        batch_size = 1000
        for start_idx in range(0, self.source_len, batch_size):
            end_idx = min(start_idx + batch_size, self.source_len)
            if start_idx % 10000 == 0:
                print(f"Verarbeite Samples {start_idx} bis {end_idx}...")
                
            # Batch-Verarbeitung der Labels
            batch_labels = [self.dataset.source_dataset[i]['labels'] for i in range(start_idx, end_idx)]
            batch_labels = torch.stack(batch_labels)
            
            # Effiziente Zuordnung zu Klassen
            for class_idx in range(len(self.class_weights)):
                class_present = batch_labels[:, class_idx] > 0
                class_indices = [i + start_idx for i, present in enumerate(class_present) if present]
                indices_by_class[class_idx].extend(class_indices)
        
        # Speichere Cache
        with open(self.cache_file, 'w') as f:
            json.dump([list(map(int, indices)) for indices in indices_by_class], f)
            
        print("Cache erstellt und gespeichert.")
        return indices_by_class

    def _sample_source_indices(self, num_samples):
        """Effizientes Sampling basierend auf Klassengewichten"""
        samples_per_class = (num_samples * self.class_weights).round().int()
        sampled_indices = []
        
        # Sampling für jede Klasse
        for class_idx, n_samples in enumerate(samples_per_class):
            if len(self.source_indices_by_class[class_idx]) > 0:
                sampled_indices.extend(
                    random.choices(self.source_indices_by_class[class_idx], k=n_samples.item())
                )
        
        # Korrigiere Länge falls nötig
        while len(sampled_indices) != num_samples:
            if len(sampled_indices) < num_samples:
                class_idx = random.choices(range(len(self.class_weights)), 
                                        weights=self.class_weights, k=1)[0]
                sampled_indices.append(random.choice(self.source_indices_by_class[class_idx]))
            else:
                sampled_indices.pop()
                
        return sampled_indices

    def __iter__(self):
        # Sample Source-Indices mit Klassengewichtung
        source_indices = self._sample_source_indices(
            self.num_batches * (self.batch_size // 2)
        )
        
        # Sample Target-Indices
        target_indices = random.sample(
            self.target_indices,
            self.num_batches * (self.batch_size // 2)
        )
        
        # Erstelle Batches
        for i in range(self.num_batches):
            start_idx = i * (self.batch_size // 2)
            end_idx = start_idx + (self.batch_size // 2)
            
            batch = (source_indices[start_idx:end_idx] + 
                    target_indices[start_idx:end_idx])
            random.shuffle(batch)
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
                'labels': sample['labels'][:5],
                'domain': torch.tensor(1),
                'video': sample['video'],
                'frame': sample['frame']
            })
        
        return base_sample

def balanced_dataloader(split='train', batch_size=32):
    """
    Creates a balanced DataLoader for training, validation or testing
    
    Args:
        split (str): 'train', 'val' or 'test'
        batch_size (int): Batch size (must be even for domain balancing)
    """
    CONFIG = {
        'source_dir': Path("/data/Bartscht/CholecT50"),
        'target_dir': Path(f"/data/Bartscht/HeiChole/domain_adaptation/{split}"),
        'batch_size': batch_size,
        'cache_dir': Path(f"/data/Bartscht/cache/{split}")
    }
    
    print(f"\nVerwendeter Cache-Pfad: {CONFIG['cache_dir']}")
    
    combined_dataset = CombinedDataset(CONFIG['source_dir'], CONFIG['target_dir'])
    
    # Use the new DomainBalancedClassWeightedBatchSampler
    balanced_sampler = BalancedBatchSampler(
        combined_dataset, 
        CONFIG['batch_size'],
        cache_dir=CONFIG['cache_dir']
    )
    
    dataloader = DataLoader(
        combined_dataset,
        batch_sampler=balanced_sampler,
        num_workers=4
    )
    
    # Debug info only for training
    if split == 'train':
        for batch in dataloader:
            print("\nFirst Batch Shapes:")
            for key, value in batch.items():
                if torch.is_tensor(value):
                    print(f"{key}: {value.shape}")
                else:
                    print(f"{key}: {type(value)} with length {len(value)}")
            break
    
    return dataloader

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataloader = balanced_dataloader()