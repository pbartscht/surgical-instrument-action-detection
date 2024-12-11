import torch
from pathlib import Path
from src.utils.custom_yolo import CustomYOLO

def main():
    # Modellpfad relativ zum Projektverzeichnis
    model = CustomYOLO('yolov8l.pt')
    
    training_config = {
        # Basis-Parameter
        'data': str(Path('config') / 'data.yaml'),  # relativer Pfad zur data.yaml
        'epochs': 120,
        'imgsz': 512,
        'batch': 8,
        
        # Loss-Gewichtungen
        'box': 4.0,
        'cls': 3.5,
        
        # Optimierung und Regularisierung
        'lr0': 0.0005,
        'lrf': 0.01,
        'weight_decay': 0.0015,
        'warmup_epochs': 3,
        'patience': 30,
        'label_smoothing': 0.15,
        
        # Rest bleibt gleich
        'amp': True,
        'cos_lr': True,
        'optimizer': 'AdamW',
        'multi_scale': True,
        'overlap_mask': True,
        'save_period': 10,
        'plots': True,
        'val': True,
        'single_cls': False,
        'rect': False
    }
    
    # Training starten
    model.train(**training_config)

if __name__ == '__main__':
    main()