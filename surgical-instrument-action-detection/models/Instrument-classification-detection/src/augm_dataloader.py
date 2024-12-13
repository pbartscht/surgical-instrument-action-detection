import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset
import albumentations as A
import torch
import os
import yaml

class BasicSurgicalYOLODataset(YOLODataset):
    """
    Custom YOLO dataset for surgical instrument detection with specialized medical augmentations.
    Extends the base YOLODataset with augmentations specifically designed for surgical scenarios.
    
    Features:
    - Basic augmentations for surgical environments (brightness, contrast, geometric)
    - Medical-specific augmentations (CLAHE for metallic instruments, shadow simulation)
    - Memory-efficient processing
    - Fallback handling for augmentation failures
    """
    
    def __init__(self, data_path, augment=False, class_weights=None):
        print(f"Initialisiere Dataset mit Pfad: {data_path}")
        
        # YAML-Konfiguration laden und als data-Attribut speichern
        # Dies ist kritisch für die korrekte Funktionsweise der YOLODataset-Klasse
        with open(data_path, 'r') as f:
            self.data = yaml.safe_load(f)
        
        # Den vollständigen Bildpfad aus der YAML-Konfiguration konstruieren
        self.img_path = os.path.join(self.data['path'], self.data['train'])
        print(f"Verwende Bildpfad: {self.img_path}")
        
        # Die Originalkonfiguration an die Elternklasse übergeben
        # Wichtig: Wir übergeben data_path, nicht self.img_path
        super().__init__(data_path, augment=augment)
        self.class_weights = class_weights

        if augment:
            # Stage 1: Essential base augmentations for surgical instruments
            basic_transforms = [
                # Brightness and contrast adjustment for varying OR lighting conditions
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.2),
                    contrast_limit=(-0.2, 0.2),
                    p=0.8  # High probability as it's crucial for OR scenes
                ),
                
                # Geometric transformations for various instrument positions
                A.ShiftScaleRotate(
                    shift_limit=0.1,    # Subtle position shifts
                    scale_limit=0.1,    # Scaling for depth variations
                    rotate_limit=20,    # Rotation for different grip angles
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.7
                ),
            ]

            # Stage 2: Medical-specific augmentations
            medical_transforms = [
                # CLAHE for better instrument contours
                A.CLAHE(
                    clip_limit=2.0,
                    tile_grid_size=(4, 4),
                    p=0.3
                ),
                
                # Shadow simulation for realistic OR conditions
                A.RandomShadow(
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=3,
                    p=0.3
                ),
            ]

            # Combine all transformations
            self.instrument_aug = A.Compose(
                basic_transforms + medical_transforms,
                bbox_params=A.BboxParams(
                    format='yolo',
                    label_fields=['class_labels']
                )
            )
            print("Augmentierungen erfolgreich initialisiert")
        else:
            self.instrument_aug = None
            print("Augmentierungen deaktiviert")

    def __getitem__(self, index):
        """
        Lädt und verarbeitet ein einzelnes Bild-Label-Paar aus dem Dataset.
        Führt Augmentierungen durch, wenn diese aktiviert sind.
        """
        # Original-Bild und Labels von der Elternklasse holen
        im, labels = super().__getitem__(index)

        if self.augment and self.instrument_aug is not None:
            try:
                # Konvertiere Labels in das richtige Format für Albumentations
                bboxes = labels[:, 1:].tolist()  # YOLO-Format: [x_center, y_center, width, height]
                class_labels = labels[:, 0].tolist()  # Klassenindizes

                # Führe Augmentierungen durch
                transformed = self.instrument_aug(
                    image=im,
                    bboxes=bboxes,
                    class_labels=class_labels
                )

                # Erstelle neue Labels-Matrix mit augmentierten Bounding Boxes
                if transformed['bboxes']:
                    new_boxes = np.array(transformed['bboxes'])
                    new_labels = np.array(transformed['class_labels'])
                    labels = np.column_stack((new_labels, new_boxes))

                return transformed['image'], labels
            except Exception as e:
                print(f"Warnung: Fehler bei Augmentierung für Bild {index}: {str(e)}")
                return im, labels  # Fallback auf nicht-augmentierte Daten
        
        return im, labels