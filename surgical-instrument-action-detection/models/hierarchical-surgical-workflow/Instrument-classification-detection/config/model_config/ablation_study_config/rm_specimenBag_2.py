from pathlib import Path
import os

def remove_empty_labels_and_images(labels_dir, images_dir):
    # Convert to Path objects
    labels_base = Path(labels_dir)
    images_base = Path(images_dir)
    
    # Counter für Statistiken
    empty_files_count = 0
    deleted_labels = 0
    deleted_images = 0
    
    # Suche in allen Unterordnern (train, val, test)
    subfolders = ['train', 'val', 'test']
    
    for subfolder in subfolders:
        labels_path = labels_base / subfolder
        images_path = images_base / subfolder
        
        print(f"\nProcessing {subfolder} folder...")
        
        # Finde alle leeren txt Dateien
        empty_files = [f for f in labels_path.glob('*.txt') if f.stat().st_size == 0]
        empty_files_count += len(empty_files)
        
        for empty_file in empty_files:
            # Lösche leere Label-Datei
            empty_file.unlink()
            deleted_labels += 1
            
            # Finde und lösche entsprechendes Bild
            image_stem = empty_file.stem
            # Suche nach allen gängigen Bildformaten
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_file = images_path / f"{image_stem}{ext}"
                if image_file.exists():
                    image_file.unlink()
                    deleted_images += 1
                    break
    
    # Ausgabe Statistik
    print("\nDeletion Summary:")
    print(f"Found {empty_files_count} empty label files")
    print(f"Deleted {deleted_labels} label files")
    print(f"Deleted {deleted_images} corresponding images")

if __name__ == "__main__":
    # Pfade anpassen
    labels_dir = "/data/Bartscht/YOLO1/labels_no_specimen"
    images_dir = "/data/Bartscht/YOLO1/images"
    
    # Sicherheitsabfrage
    print("This will delete empty label files and their corresponding images.")
    print(f"Labels directory: {labels_dir}")
    print(f"Images directory: {images_dir}")
    remove_empty_labels_and_images(labels_dir, images_dir)
    print("\nDeletion completed!")