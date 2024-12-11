import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import pytorch_lightning as pl
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

class VerbDataset(Dataset):
    """
    Dataset für chirurgische Verb-Klassifikation mit Instrumenten-Information
    Gibt nun den Instrumentennamen statt des Index zurück
    """
    def __init__(self, base_dir: str, transform: Optional[transforms.Compose] = None, phase: str = 'train'):
        self.base_dir = base_dir
        self.transform = transform
        self.phase = phase
        
        # Definiere Verb- und Instrument-Klassen
        self.verb_names = ["dissect", "retract", "null_verb", "coagulate", "grasp", 
                          "clip", "aspirate", "cut", "irrigate", "pack"]
        self.instrument_names = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"]
        
        # Erstelle Mapping-Dictionaries
        self.verb_to_idx = {verb: idx for idx, verb in enumerate(self.verb_names)}
        
        # Lade die Daten
        self.data = self.load_data()
        
        # Erstelle Mapping von Verb zu erlaubten Instrumenten
        self.verb_instrument_constraints = self._create_verb_instrument_constraints()
    
    def _create_verb_instrument_constraints(self) -> Dict[str, List[str]]:
        """
        Erstellt ein Dictionary mit den erlaubten Instrumenten für jedes Verb
        basierend auf den tatsächlichen Daten
        """
        constraints = {}
        for verb in self.verb_names:
            valid_instruments = set(
                item['instrument'] for item in self.data 
                if item['verb'] == verb
            )
            constraints[verb] = list(valid_instruments)
        return constraints

    def load_data(self) -> List[Dict]:
        """
        Lädt die Daten aus der Verzeichnisstruktur
        """
        all_data = []
        
        for vid_dir in tqdm(os.listdir(self.base_dir), desc="Loading data"):
            if not vid_dir.startswith('VID'):
                continue
                
            csv_path = os.path.join(self.base_dir, vid_dir, 'labels.csv')
            if not os.path.exists(csv_path):
                print(f"Warnung: Keine labels.csv in {vid_dir} gefunden")
                continue
                
            try:
                df = pd.read_csv(csv_path)
                
                for _, row in df.iterrows():
                    if row['Verb'] in self.verb_names and row['Instrument'] in self.instrument_names:
                        img_path = os.path.join(self.base_dir, vid_dir, row['Dateiname'])
                        
                        if os.path.exists(img_path):
                            all_data.append({
                                'img_path': img_path,
                                'verb': row['Verb'],
                                'instrument': row['Instrument'],
                                'confidence': row['Confidence'],
                                'frame': row['Frame'],
                                'vid': vid_dir,
                                'filename': row['Dateiname']
                            })
                        
            except Exception as e:
                print(f"Fehler beim Laden von {csv_path}: {e}")
        
        if not all_data:
            raise RuntimeError("Keine Daten geladen!")
            
        print(f"Insgesamt {len(all_data)} Bilder geladen")
        return all_data

    def get_verb_labels(self) -> List[int]:
        """Gibt alle Verb-Labels zurück"""
        return [self.verb_to_idx[item['verb']] for item in self.data]

    def get_instruments(self) -> List[str]:
        """Gibt alle Instrumentennamen zurück"""
        return [item['instrument'] for item in self.data]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int, float]:
        """
        Gibt ein Tupel zurück: (Bild, Instrumentenname, Verb-Label, Confidence)
        Instrumentenname wird als String übergeben für Constraint-Verarbeitung
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} außerhalb des gültigen Bereichs (0-{len(self.data)-1})")
            
        img_data = self.data[idx]
        try:
            # Lade Bild
            image = Image.open(img_data['img_path']).convert('RGB')
            
            # Hole Labels
            verb_label = self.verb_to_idx[img_data['verb']]
            instrument_name = img_data['instrument']  # String statt Index
            confidence = img_data['confidence']

            if self.transform:
                image = self.transform(image)

            return image, instrument_name, verb_label
            
        except Exception as e:
            raise Exception(f"Fehler beim Laden von Bild {img_data['img_path']}: {e}")

    def get_verb_instrument_constraints(self) -> Dict[str, List[str]]:
        """
        Gibt das Dictionary mit Verb-Instrument-Constraints zurück
        """
        return self.verb_instrument_constraints

class VerbDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule für das Verb-Dataset
    """
    def __init__(self, base_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.full_dataset = VerbDataset(
                self.base_dir, 
                self.train_transform, 
                'train'
            )
            
            dataset_size = len(self.full_dataset)
            indices = list(range(dataset_size))
            
            train_size = int(0.85 * dataset_size)
            
            np.random.shuffle(indices)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            self.train_dataset = Subset(self.full_dataset, train_indices)
            self.val_dataset = Subset(self.full_dataset, val_indices)
            
            # Berechne Gewichte basierend auf Verb-Instrument-Kombinationen
            verb_instrument_pairs = [
                (self.full_dataset.data[i]['verb'], self.full_dataset.data[i]['instrument'])
                for i in train_indices
            ]
            pair_counts = Counter(verb_instrument_pairs)
            
            total_samples = len(verb_instrument_pairs)
            weights = [
                total_samples / (len(pair_counts) * pair_counts[pair])
                for pair in verb_instrument_pairs
            ]
            self.train_weights = torch.DoubleTensor(weights)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=WeightedRandomSampler(
                weights=self.train_weights,
                num_samples=len(self.train_dataset),
                replacement=True
            ),
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_verb_instrument_constraints(self) -> Dict[str, List[str]]:
        """
        Gibt die Verb-Instrument-Constraints zurück
        """
        return self.full_dataset.get_verb_instrument_constraints()

# Beispiel für die Verwendung:
if __name__ == "__main__":
    # Test des Dataloaders
    base_dir = "/data/Bartscht/Verbs"
    data_module = VerbDataModule(base_dir)
    data_module.setup()
    
    # Hole die Constraints
    constraints = data_module.get_verb_instrument_constraints()
    print("\nVerb-Instrument Constraints:")
    for verb, instruments in constraints.items():
        print(f"{verb}: {instruments}")
    
    # Teste den Dataloader
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    images, instrument_names, verb_labels, confidences = batch
    
    print("\nBeispiel-Batch:")
    print(f"Bilder Shape: {images.shape}")
    print(f"Instrument Namen (erste 5): {instrument_names[:5]}")
    print(f"Verb Labels (erste 5): {verb_labels[:5]}")
    print(f"Confidences (erste 5): {confidences[:5]}")