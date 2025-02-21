from pathlib import Path

def convert_all_classes_to_seven(base_dir):
    # Counter für Statistik
    total_files = 0
    total_labels_converted = 0
    
    # Unterordner
    subfolders = ['train', 'val', 'test']
    
    print("Starting conversion process...")
    
    for subfolder in subfolders:
        print(f"\nProcessing {subfolder} folder...")
        labels_path = Path(base_dir) / subfolder
        
        if not labels_path.exists():
            print(f"Folder {subfolder} not found, skipping...")
            continue
        
        # Verarbeite alle txt Dateien
        files_processed = 0
        labels_in_folder = 0
        
        for txt_file in labels_path.glob('*.txt'):
            modified_lines = []
            
            # Lese Datei
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            # Konvertiere jede Zeile
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:  # Prüfe ob Format korrekt (class + 4 coordinates)
                    # Ersetze Klassenindex mit 7
                    new_line = f"7 {' '.join(parts[1:])}\n"
                    modified_lines.append(new_line)
                    labels_in_folder += 1
            
            # Schreibe modifizierte Zeilen zurück
            with open(txt_file, 'w') as f:
                f.writelines(modified_lines)
            
            files_processed += 1
            
        print(f"Processed {files_processed} files in {subfolder}")
        print(f"Converted {labels_in_folder} labels in {subfolder}")
        
        total_files += files_processed
        total_labels_converted += labels_in_folder
    
    # Gesamtstatistik
    print("\nConversion Summary:")
    print(f"Total files processed: {total_files}")
    print(f"Total labels converted to class 7: {total_labels_converted}")

if __name__ == "__main__":
    # Pfad zum labels Ordner
    labels_dir = "/data/Bartscht/YOLO1/labels"
    
    print(f"Converting all class indices to 7 in: {labels_dir}")
    convert_all_classes_to_seven(labels_dir)
    print("\nConversion completed!")