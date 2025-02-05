import os
import shutil
from pathlib import Path
import yaml
import random

def setup_dataset_structure(base_path, train_ratio=0.8, val_ratio=0.1):
    """
    Reorganisiert die mixed_samples in eine YOLO-kompatible Struktur
    """
    base_path = Path(base_path)
    
    # Erstelle neue Verzeichnisstruktur
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            (base_path / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # Liste alle vorhandenen Bilder
    existing_images = list((base_path / "images").glob("*.png"))
    random.shuffle(existing_images)
    
    # Berechne Split-Indices
    n_images = len(existing_images)
    n_train = int(n_images * train_ratio)
    n_val = int(n_images * val_ratio)
    
    # Verteile Bilder und Labels
    splits = {
        'train': existing_images[:n_train],
        'val': existing_images[n_train:n_train + n_val],
        'test': existing_images[n_train + n_val:]
    }
    
    for split, images in splits.items():
        for img_path in images:
            # Kopiere Bild
            new_img_path = base_path / split / "images" / img_path.name
            shutil.copy(img_path, new_img_path)
            
            # Kopiere entsprechendes Label
            label_path = base_path / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                new_label_path = base_path / split / "labels" / f"{img_path.stem}.txt"
                shutil.copy(label_path, new_label_path)

def create_dataset_yaml(base_path):
    """
    Erstellt die dataset.yaml Datei für YOLO
    """
    yaml_content = {
        'path': str(base_path),  # Basispfad
        'train': 'train/images',  # Relativer Pfad zu Trainingsbildern
        'val': 'val/images',      # Relativer Pfad zu Validierungsbildern
        'test': 'test/images',    # Relativer Pfad zu Testbildern
        
        # Klasseninformationen von deiner ursprünglichen yaml
        'nc': 7,
        'names': [
            'Grasper',
            'Bipolar',
            'Hook',
            'Scissors',
            'Clipper',
            'Irrigator',
            'SpecimenBag'
        ],
        
        # Klassengewichte von deiner ursprünglichen yaml
        'class_weights': {
            '0': 0.32,
            '1': 1.47,
            '2': 0.64,
            '3': 3.1,
            '4': 2.36,
            '5': 1.39,
            '6': 0.0
        }
    }
    
    # Speichere yaml
    yaml_path = Path(base_path) / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

def main():
    base_path = "/data/Bartscht/mixed_samples"
    
    # 1. Setup Verzeichnisstruktur
    print("Setting up directory structure...")
    setup_dataset_structure(base_path)
    
    # 2. Erstelle dataset.yaml
    print("Creating dataset.yaml...")
    create_dataset_yaml(base_path)
    
    print("Dataset setup complete!")

if __name__ == "__main__":
    main()