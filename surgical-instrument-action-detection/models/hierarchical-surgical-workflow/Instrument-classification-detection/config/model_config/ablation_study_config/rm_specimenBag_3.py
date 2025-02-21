from pathlib import Path
import shutil
import os

def validate_and_copy_datasets(labels_dir, images_dir, output_labels_dir, output_images_dir):
    # Counter für Statistik
    total_valid_pairs = 0
    total_missing_images = 0
    
    # Unterordner
    subfolders = ['train', 'val', 'test']
    
    for subfolder in subfolders:
        print(f"\nProcessing {subfolder} folder...")
        
        # Setup Pfade
        labels_path = Path(labels_dir) / subfolder
        images_path = Path(images_dir) / subfolder
        out_labels_path = Path(output_labels_dir) / subfolder
        out_images_path = Path(output_images_dir) / subfolder
        
        # Erstelle Ausgabeordner
        out_labels_path.mkdir(parents=True, exist_ok=True)
        out_images_path.mkdir(parents=True, exist_ok=True)
        
        # Verarbeite alle Label-Dateien
        valid_pairs = 0
        missing_images = 0
        
        for label_file in labels_path.glob('*.txt'):
            # Überspringe leere Label-Dateien
            if label_file.stat().st_size == 0:
                continue
                
            # Suche entsprechendes Bild
            image_found = False
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                image_file = images_path / f"{label_file.stem}{ext}"
                if image_file.exists():
                    # Kopiere validiertes Paar
                    shutil.copy2(label_file, out_labels_path / label_file.name)
                    shutil.copy2(image_file, out_images_path / image_file.name)
                    valid_pairs += 1
                    image_found = True
                    break
            
            if not image_found:
                missing_images += 1
                print(f"Missing image for label: {label_file.name}")
        
        print(f"Found {valid_pairs} valid image-label pairs in {subfolder}")
        print(f"Missing {missing_images} images in {subfolder}")
        
        total_valid_pairs += valid_pairs
        total_missing_images += missing_images
    
    # Gesamtstatistik
    print("\nFinal Summary:")
    print(f"Total valid image-label pairs: {total_valid_pairs}")
    print(f"Total missing images: {total_missing_images}")
    print(f"\nValidated datasets copied to:")
    print(f"Labels: {output_labels_dir}")
    print(f"Images: {output_images_dir}")

if __name__ == "__main__":
    # Pfade anpassen
    labels_dir = "/data/Bartscht/YOLO1/labels_no_specimen"  # Dein gefilterter Labels-Ordner
    images_dir = "/data/Bartscht/YOLO1/images"  # Original Bilder-Ordner
    output_labels_dir = "/data/Bartscht/YOLO1/labels_corrected"  # Neuer Labels-Ordner
    output_images_dir = "/data/Bartscht/YOLO1/images_corrected"  # Neuer Bilder-Ordner
    
    validate_and_copy_datasets(labels_dir, images_dir, output_labels_dir, output_images_dir)