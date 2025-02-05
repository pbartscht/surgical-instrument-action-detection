import yaml
from pathlib import Path
from ultralytics import YOLO

def create_dataset_yaml(mixed_samples_path):
    """
    Erstellt eine YAML-Konfigurationsdatei für das Training
    """
    dataset_config = {
        'path': mixed_samples_path,  # Pfad zum Hauptverzeichnis
        'train': {
            'path': 'images'  # Relativer Pfad zu Trainingsbildern
        },
        'val': {
            'path': 'images'  # Für Transfer Learning erstmal gleiche Bilder
        },
        'names': {
            0: 'grasper',
            1: 'bipolar', 
            2: 'hook',
            3: 'scissors',
            4: 'clipper',
            5: 'irrigator'
        },
        'nc': 6  # Anzahl der Klassen
    }
    
    # YAML-Datei speichern
    yaml_path = Path(mixed_samples_path) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(dataset_config, f, sort_keys=False)
    
    return yaml_path

def setup_transfer_learning(pretrained_model_path, mixed_samples_path, project_name="yolo_transfer"):
    """
    Konfiguriert und startet das Transfer Learning
    """
    # YAML-Konfiguration erstellen
    yaml_path = create_dataset_yaml(mixed_samples_path)
    
    # Vortrainiertes Modell laden
    model = YOLO(pretrained_model_path)
    
    # Trainingsparameter für Transfer Learning
    training_args = {
        'data': str(yaml_path),          # Pfad zur YAML-Konfiguration
        'epochs': 50,                    # Anzahl der Epochen
        'imgsz': 640,                    # Bildgröße
        'batch': 16,                     # Batch-Größe
        'device': 0,                     # GPU (falls verfügbar)
        'workers': 8,                    # Anzahl der Datenlade-Prozesse
        'patience': 15,                  # Early Stopping Geduld
        
        # Learning Rate Parameter
        'lr0': 0.0001,                  # Niedrigere initiale Learning Rate für Transfer Learning
        'lrf': 0.01,                    # Finale Learning Rate als Faktor von lr0
        'warmup_epochs': 3.0,           # Warm-up Epochen
        'warmup_momentum': 0.8,         # Warm-up Momentum
        
        # Optimizer Settings
        'optimizer': 'Adam',            # Adam Optimizer
        'weight_decay': 0.0005,         # L2 Regularisierung
        'momentum': 0.937,              # Momentum für Optimizer
        
        # Weitere Parameter
        'cos_lr': True,                 # Cosine Learning Rate Scheduling
        'resume': False,                # Nicht von letztem Checkpoint fortsetzen
        'exist_ok': True,               # Überschreibe existierende Experimente
        'pretrained': True,             # Nutze vortrainierte Gewichte
        'amp': True,                    # Automatisches Mixed Precision Training
        
        # Projekt Einstellungen
        'project': project_name,        # Projektname für die Logs
        'name': 'transfer_learning',    # Name des Experiments
        'save_period': 10,             # Speichere Checkpoints alle 10 Epochen
    }
    
    # Training starten
    results = model.train(**training_args)
    return results

def main():
    # Pfade definieren
    PRETRAINED_MODEL = "/data/Bartscht/YOLO/best_v35.pt"
    MIXED_SAMPLES_PATH = "/data/Bartscht/mixed_samples"
    
    # Transfer Learning starten
    results = setup_transfer_learning(
        PRETRAINED_MODEL, 
        MIXED_SAMPLES_PATH,
        project_name="heichole_transfer"
    )
    
    print("Training abgeschlossen!")
    print(f"Ergebnisse: {results}")

if __name__ == "__main__":
    main()