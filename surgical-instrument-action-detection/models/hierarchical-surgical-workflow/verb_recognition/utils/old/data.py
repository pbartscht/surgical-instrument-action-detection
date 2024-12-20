import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import List, Dict, Tuple

class VerbDataset(Dataset):
    def __init__(self, base_dir: str, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.verb_names = ["grasp", "retract", "dissect", "coagulate", "clip", "cut", "aspirate", "irrigate", "pack", "null_verb"]
        self.class_to_idx = {verb: idx for idx, verb in enumerate(self.verb_names)}
        self.data = self.load_data()
        print(f"Classes: {self.verb_names}")
        print(f"Class to index mapping: {self.class_to_idx}")

    def load_data(self) -> List[Dict[str, str]]:
        data = []
        print(f"Base directory: {self.base_dir}")
        
        for vid_folder in os.listdir(self.base_dir):
            if vid_folder.startswith("VID"):
                print(f"Processing folder: {vid_folder}")
                img_dir = os.path.join(self.base_dir, vid_folder)
                csv_file = os.path.join(self.base_dir, "results", f"{vid_folder}_image_verbs.csv")
                
                print(f"Image directory: {img_dir}")
                print(f"CSV file: {csv_file}")
                
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file, sep='|', names=['Dateiname', 'Verb'])
                    file_verb_map = dict(zip(df['Dateiname'], df['Verb']))
                    
                    for img_file in os.listdir(img_dir):
                        if img_file.endswith('.png'):
                            if img_file in file_verb_map:
                                verb = file_verb_map[img_file]
                                if verb in self.verb_names:
                                    data.append({
                                        'img_path': os.path.join(img_dir, img_file),
                                        'verb': verb
                                    })
                                    print(f"Added image: {img_file} with verb: {verb}")
                                else:
                                    print(f"Unrecognized verb '{verb}' for image: {img_file}")
                            else:
                                print(f"Image not found in CSV: {img_file}")
                else:
                    print(f"CSV file not found: {csv_file}")

        print(f"Total data points loaded: {len(data)}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        image = Image.open(img_data['img_path']).convert('RGB')
        label = self.class_to_idx[img_data['verb']]

        if self.transform:
            image = self.transform(image)

        return image, label

class VerbDataModule(pl.LightningDataModule):
    def __init__(self, base_dir: str, batch_size: int):
        super().__init__()
        self.base_dir = base_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        full_dataset = VerbDataset(self.base_dir, self.transform)

        # Splitting the dataset
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

def get_num_classes(base_dir: str) -> int:
    dataset = VerbDataset(base_dir)
    return len(dataset.verb_names)

if __name__ == "__main__":
    # Pfade für den Test
    base_dir = "/data/Bartscht/cropped_images"
    batch_size = 32

    # Datamodule erstellen
    data_module = VerbDataModule(base_dir, batch_size)
    data_module.setup()

    # Informationen ausgeben
    print(f"Anzahl der Klassen: {get_num_classes(base_dir)}")
    print(f"Größe des Trainingsdatensatzes: {len(data_module.train_dataset)}")
    print(f"Größe des Validierungsdatensatzes: {len(data_module.val_dataset)}")
    print(f"Größe des Testdatensatzes: {len(data_module.test_dataset)}")

    # Ein paar Beispiele aus dem Trainingsdatensatz anzeigen
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    images, labels = batch

    print(f"Batch-Größe: {images.shape[0]}")
    print(f"Bildgröße: {images.shape[1:]}")
    print(f"Label-Beispiele: {labels[:5]}")

    # Optional: Überprüfen von Bildern und deren Labels
    dataset = data_module.train_dataset.dataset
    for i in range(5):
        img, label = dataset[i]
        print(f"Bild {i}: Shape {img.shape}, Label {label}")