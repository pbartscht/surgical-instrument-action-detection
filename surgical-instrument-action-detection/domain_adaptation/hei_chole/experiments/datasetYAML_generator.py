import os
from pathlib import Path
import numpy as np
from collections import Counter
import yaml

def analyze_class_distribution(labels_dir):
    """
    Analysiert die Klassenverteilung in den YOLO-Labels
    """
    class_counts = Counter()
    total_objects = 0
    
    # Durchsuche alle Label-Dateien
    for label_file in Path(labels_dir).glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1
                total_objects += 1
    
    return class_counts, total_objects

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
    copy old dataset.yaml into new mixed_samples_epoch_x to refresh class distribution
    Aktualisiert die dataset.yaml mit den neuen Klassengewichten
    """
    with open(yaml_path, 'r') as f:
        yaml_content = yaml.safe_load(f)
    
    # Aktualisiere Klassengewichte
    yaml_content['class_weights'] = {str(k): float(v) for k, v in class_weights.items()}
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

def main():
    base_path = "/data/Bartscht/mixed_samples_epoch3"
    labels_dir = os.path.join(base_path, "labels")
    yaml_path = os.path.join(base_path, "dataset.yaml")
    
    print("Analyzing class distribution...")
    class_counts, total_objects = analyze_class_distribution(labels_dir)
    
    print("\nClass distribution:")
    for class_id, count in sorted(class_counts.items()):
        percentage = (count / total_objects) * 100
        print(f"Class {class_id}: {count} objects ({percentage:.2f}%)")
    
    print("\nCalculating new class weights...")
    class_weights = calculate_class_weights(class_counts, total_objects, method='inverse')
    
    print("\nNew class weights:")
    for class_id, weight in sorted(class_weights.items()):
        print(f"Class {class_id}: {weight:.3f}")
    
    print("\nUpdating dataset.yaml...")
    update_dataset_yaml(yaml_path, class_weights)
    
    print("Done! Dataset YAML updated with new class weights.")

if __name__ == "__main__":
    main()