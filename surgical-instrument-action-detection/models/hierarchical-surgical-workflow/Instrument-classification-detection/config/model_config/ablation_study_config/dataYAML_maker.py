import os
from pathlib import Path
import numpy as np
from collections import Counter
import yaml

def analyze_class_distribution(labels_dir):
    """
    Analysiert die Klassenverteilung in den YOLO-Label-Dateien.
    Überspringt Dateien, die ungültige Zeilen enthalten.
    """
    class_counts = Counter()
    total_objects = 0
    num_files = 0
    
    # Durchsuche alle .txt-Dateien im labels_dir
    for label_file in Path(labels_dir).glob("*.txt"):
        num_files += 1
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # Überprüfe, ob das erste Token eine Zahl ist
                if parts[0].isdigit():
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    total_objects += 1
                else:
                    # Zeile enthält keine gültigen Label-Daten, überspringen
                    continue
                    
    return class_counts, total_objects, num_files

def calculate_class_weights(class_counts, total_objects, method='inverse'):
    """
    Berechnet Klassengewichte basierend auf der Verteilung
    
    Methoden:
    - 'inverse': 1 / frequency (mit Normalisierung)
    - 'balanced': N_samples / (n_classes * n_samples_class)
    """
    weights = {}
    n_classes = len(class_counts)
    
    if method == 'inverse':
        # Inverse frequency weighting
        max_count = max(class_counts.values())
        for class_id, count in class_counts.items():
            weights[class_id] = max_count / (count + 1e-6)  # Verhindere Division durch 0
            
    elif method == 'balanced':
        # Balanced weighting
        for class_id, count in class_counts.items():
            weights[class_id] = total_objects / (n_classes * count + 1e-6)
    
    # Normalisiere Gewichte
    max_weight = max(weights.values())
    for class_id in weights:
        weights[class_id] = weights[class_id] / max_weight
    
    return weights

def update_dataset_yaml(yaml_path, class_weights):
    """
    Aktualisiert die dataset.yaml mit den neuen Klassengewichten
    """
    with open(yaml_path, 'r') as f:
        yaml_content = yaml.safe_load(f)
    
    # Aktualisiere Klassengewichte
    yaml_content['class_weights'] = {str(k): float(v) for k, v in class_weights.items()}
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

def print_distribution(split_name, class_counts, total_objects, num_files):
    print(f"\n=== {split_name} Set Distribution ===")
    print(f"Total files: {num_files}")
    print(f"Total objects: {total_objects}")
    print("\nPer-class distribution:")
    for class_id, count in sorted(class_counts.items()):
        percentage = (count / total_objects) * 100
        print(f"Class {class_id}: {count} objects ({percentage:.2f}%)")

def main():
    base_path = "/data/Bartscht/YOLO"
    train_dir = os.path.join(base_path, "labels", "train")
    val_dir = os.path.join(base_path, "labels", "val")
    yaml_path = os.path.join(base_path, "data.yaml")
    
    # Analyze training set
    print("Analyzing training set distribution...")
    train_counts, train_total, train_files = analyze_class_distribution(train_dir)
    print_distribution("Training", train_counts, train_total, train_files)
    
    # Analyze validation set
    print("\nAnalyzing validation set distribution...")
    val_counts, val_total, val_files = analyze_class_distribution(val_dir)
    print_distribution("Validation", val_counts, val_total, val_files)
    
    # Calculate weights based on training set
    print("\nCalculating new class weights (based on training set)...")
    class_weights = calculate_class_weights(train_counts, train_total, method='inverse')
    
    print("\nNew class weights:")
    for class_id, weight in sorted(class_weights.items()):
        print(f"Class {class_id}: {weight:.3f}")
    
    # Update yaml
    print("\nUpdating dataset.yaml...")
    update_dataset_yaml(yaml_path, class_weights)
    
    print("Done! Dataset YAML updated with new class weights.")

if __name__ == "__main__":
    main()