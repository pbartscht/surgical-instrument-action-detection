import os
from pathlib import Path
import shutil

def remove_specimen_bag_labels(src_base_dir, dst_base_dir):
    # Unterordner, die wir verarbeiten wollen
    subfolders = ['train', 'val', 'test']
    total_processed = 0
    
    for subfolder in subfolders:
        src_dir = Path(src_base_dir) / subfolder
        dst_dir = Path(dst_base_dir) / subfolder
        
        # Erstelle Zielordner
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        # Verarbeite alle txt Dateien im Unterordner
        txt_files = list(src_dir.glob('*.txt'))
        
        print(f"Processing {subfolder} folder...")
        for txt_file in txt_files:
            # Lese alle Zeilen
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            # Filtere SpecimenBag (Klasse 6) heraus
            filtered_lines = [line for line in lines if not line.startswith('6 ')]
            
            # Schreibe gefilterte Zeilen in neue Datei
            dst_file = dst_dir / txt_file.name
            with open(dst_file, 'w') as f:
                f.writelines(filtered_lines)
            
            # Wenn keine Labels übrig bleiben, erstelle leere Datei
            if not filtered_lines:
                dst_file.touch()
        
        total_processed += len(txt_files)
        print(f"Processed {len(txt_files)} files in {subfolder}")
    
    print(f"\nTotal files processed: {total_processed}")

# Pfade anpassen
src_path = "/data/Bartscht/YOLO1/labels"  # Originaler labels Ordner
dst_path = "/data/Bartscht/YOLO1/labels_no_specimen"  # Neuer labels Ordner

remove_specimen_bag_labels(src_path, dst_path)