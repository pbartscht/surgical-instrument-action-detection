import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset
import albumentations as A
import torch
import yaml
from pathlib import Path

class BasicSurgicalYOLODataset(YOLODataset):
    """
    Custom YOLO dataset for surgical instrument detection with specialized medical augmentations.
    """
    def __init__(self, data_path, augment=False, class_weights=None):
        # YAML-Datei laden
        with open(data_path, 'r') as f:
            data_dict = yaml.safe_load(f)
        
        # Absoluten Pfad zu den Trainingsbildern konstruieren
        base_path = Path(data_dict.get('path', ''))  # /data/Bartscht/YOLO
        train_path = base_path / data_dict.get('train', '')  # Kombiniert zu /data/Bartscht/YOLO/images/train
        
        print(f"Verwende Bilderpfad: {train_path}")
        
        # YOLO's eigene Initialisierung mit dem absoluten Bildpfad nutzen
        super().__init__(str(train_path), augment=augment)
        self.class_weights = class_weights
        
        if augment:
            self._setup_augmentations()
        else:
            self.instrument_aug = None
            print("Augmentationen deaktiviert")

    def _setup_augmentations(self):
        """Setup der Augmentations-Pipeline"""
        # Basis-Transformationen für OP-Szenen
        basic_transforms = [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.2, 0.2),
                p=0.8
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=20,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.7
            ),
        ]
        
        # Medizinspezifische Transformationen
        medical_transforms = [
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(4, 4),
                p=0.3
            ),
            A.RandomShadow(
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=3,
                p=0.3
            ),
        ]
        
        # Kombiniere alle Transformationen
        self.instrument_aug = A.Compose(
            basic_transforms + medical_transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            )
        )
        print("Augmentationen initialisiert")

    def __getitem__(self, index):
        """Lädt ein Bild und wendet Augmentationen an, falls aktiviert"""
        # Originalbild und Labels von YOLO laden
        img, labels = super().__getitem__(index)
        
        # Augmentationen nur anwenden, wenn aktiviert
        if self.augment and self.instrument_aug is not None:
            try:
                # Labels für Albumentations vorbereiten
                bboxes = labels[:, 1:].tolist()  # YOLO format: [x_center, y_center, width, height]
                class_labels = labels[:, 0].tolist()
                
                # Augmentationen durchführen
                transformed = self.instrument_aug(
                    image=img,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                # Neue Labels-Matrix erstellen
                if transformed['bboxes']:
                    new_boxes = np.array(transformed['bboxes'])
                    new_labels = np.array(transformed['class_labels'])
                    labels = np.column_stack((new_labels, new_boxes))
                
                return transformed['image'], labels
                
            except Exception as e:
                print(f"Warnung: Augmentationsfehler bei Bild {index}: {str(e)}")
                return img, labels  # Fallback auf nicht-augmentierte Daten
        
        return img, labels