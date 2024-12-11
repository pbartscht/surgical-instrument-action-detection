import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset
import albumentations as A

class BasicSurgicalYOLODataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Stufe 1: Essenzielle Basis-Augmentierungen für chirurgische Instrumente
        basic_transforms = [
            # Helligkeits- und Kontrastadjustierung für OP-Beleuchtungsvariationen
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.2, 0.2),
                p=0.8  # Hohe Wahrscheinlichkeit, da wichtig für OP-Szenen
            ),
            
            # Grundlegende geometrische Transformationen für verschiedene Instrumentenpositionen
            A.ShiftScaleRotate(
                shift_limit=0.1,    # Leichte Verschiebung
                scale_limit=0.1,    # Skalierung für Tiefenvariationen
                rotate_limit=20,    # Rotation für verschiedene Greifwinkel
                border_mode=cv2.BORDER_CONSTANT,
                p=0.7
            ),
        ]
        
        # Stufe 2: Medizinisch relevante Augmentierungen
        medical_transforms = [
            # CLAHE für bessere Instrumentenkonturen
            # Wichtig für metallische Instrumente gegen OP-Licht
            A.CLAHE(
                clip_limit=2.0, 
                tile_grid_size=(4, 4), 
                p=0.3
            ),
            
            # Schatten-Simulation für realistische OP-Bedingungen
            A.RandomShadow(
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=3,
                p=0.3
            ),
        ]
        
        # Kombiniere die Transformationen
        self.instrument_aug = A.Compose(
            basic_transforms + medical_transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            )
        )

    def __getitem__(self, index):
        try:
            im, labels = super().__getitem__(index)
            
            if self.augment and len(labels):
                # Speichereffiziente Verarbeitung
                augmented = self.instrument_aug(
                    image=im,
                    bboxes=labels[:, 1:].tolist(),
                    class_labels=labels[:, 0].tolist()
                )
                
                im = augmented['image']
                if augmented['bboxes']:
                    labels = np.array(
                        [[c, *b] for c, b in zip(
                            augmented['class_labels'],
                            augmented['bboxes']
                        )],
                        dtype=np.float32
                    )
                
                # Explizit Speicher freigeben
                del augmented
                
            return im, labels

        except Exception as e:
            print(f"Augmentation error at index {index}: {e}")
            # Fallback: Gebe unaugmentiertes Bild zurück
            return super().__getitem__(index)