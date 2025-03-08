import os
import glob

# Pfad zu Ihrem Label-Ordner
label_path = "/data/Bartscht/YOLO1/labels/val"  # Passen Sie den Pfad an

# Alle .txt Dateien finden
label_files = glob.glob(os.path.join(label_path, "*.txt"))

for file in label_files:
    # Datei lesen
    with open(file, 'r') as f:
        lines = f.readlines()
    
    # Neue Zeilen mit korrigierten Klassen
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts and parts[0] == '7':  # Wenn Klasse 7 ist
            parts[0] = '0'  # Auf 0 ändern
        new_lines.append(' '.join(parts) + '\n')
    
    # Datei mit geänderten Inhalten überschreiben
    with open(file, 'w') as f:
        f.writelines(new_lines)

print(f"Aktualisiert: {len(label_files)} Dateien")