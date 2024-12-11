import os
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
import zipfile
import torch
import numpy as np

# Pfade
MODEL_PATH = '/home/Bartscht/YOLO/runs/detect/train11/weights/best.pt'
DATASET_DIR = "/data/Bartscht/instrument_frames"  # Pfad zu frames
OUTPUT_PATH = "/home/Bartscht/YOLO/YOLO_underrepresented_instruments"

# Konfidenz-Schwellenwert und IOU-Schwellenwert für NMS
CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.3

def yolo_class_mapping():
    return {
        "grasper": 0,
        "bipolar": 1,
        "hook": 2,
        "scissors": 3,
        "clipper": 4,
        "irrigator": 5,
        "specimenBag": 6
    }

def non_max_suppression(boxes, scores, iou_threshold):
    # Convert to torch tensors
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    
    # Perform NMS
    keep = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    
    return keep.numpy()

# Laden des YOLOv11-Modells
model = YOLO(MODEL_PATH)

# Erstellen der Ordnerstruktur
TEMP_LABELS_PATH = os.path.join(OUTPUT_PATH, "obj_train_data")
os.makedirs(TEMP_LABELS_PATH, exist_ok=True)

# obj.names Datei erstellen
class_names = list(yolo_class_mapping().keys())
with open(os.path.join(OUTPUT_PATH, "obj.names"), "w") as f:
    f.write("\n".join(class_names))

# obj.data Datei erstellen
with open(os.path.join(OUTPUT_PATH, "obj.data"), "w") as f:
    f.write(f"classes = {len(class_names)}\n")
    f.write("names = obj.names\n")
    f.write("train = train.txt\n")

# train.txt Datei erstellen
train_file = open(os.path.join(OUTPUT_PATH, "train.txt"), "w")

# Iteration über alle Bilder
frame_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.png')])

for frame_file in tqdm(frame_files, desc="Processing images"):
    img_path = os.path.join(DATASET_DIR, frame_file)
    
    # Bild laden und Größe ermitteln
    with Image.open(img_path) as img:
        img_width, img_height = img.size
    
    # Vorhersage mit YOLOv8
    results = model(img_path)
    
    # Extrahieren der Bounding-Boxen, Klassen und Konfidenzen
    pred_boxes = results[0].boxes.xyxy.cpu().numpy()
    pred_classes = results[0].boxes.cls.cpu().numpy()
    pred_confidences = results[0].boxes.conf.cpu().numpy()
    
    # YOLO v11 Textdatei erstellen
    txt_filename = os.path.join(TEMP_LABELS_PATH, os.path.splitext(frame_file)[0] + '.txt')
    with open(txt_filename, "w") as txt_file:
        for instrument_class in set(pred_classes):
            class_mask = pred_classes == instrument_class
            class_boxes = pred_boxes[class_mask]
            class_confidences = pred_confidences[class_mask]
            
            if len(class_boxes) > 0:
                keep = non_max_suppression(class_boxes, class_confidences, IOU_THRESHOLD)
                
                for idx in keep:
                    if class_confidences[idx] >= CONFIDENCE_THRESHOLD:
                        box = class_boxes[idx]
                        x_center = (box[0] + box[2]) / (2 * img_width)
                        y_center = (box[1] + box[3]) / (2 * img_height)
                        width = (box[2] - box[0]) / img_width
                        height = (box[3] - box[1]) / img_height
                        txt_file.write(f"{int(instrument_class)} {x_center} {y_center} {width} {height}\n")
    
    # Pfad zur train.txt hinzufügen
    train_file.write(f"obj_train_data/{frame_file}\n")

train_file.close()

# ZIP-Datei erstellen
zip_filename = os.path.join(OUTPUT_PATH, "cvat_yolo_annotations.zip")
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    # obj.names, obj.data und train.txt hinzufügen
    zipf.write(os.path.join(OUTPUT_PATH, "obj.names"), "obj.names")
    zipf.write(os.path.join(OUTPUT_PATH, "obj.data"), "obj.data")
    zipf.write(os.path.join(OUTPUT_PATH, "train.txt"), "train.txt")
    
    # Annotationsdateien hinzufügen
    for root, dirs, files in os.walk(TEMP_LABELS_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.join("obj_train_data", file)
            zipf.write(file_path, arcname)

print(f"CVAT-kompatible YOLO-Annotationen für VID92 wurden erstellt und in {zip_filename} gespeichert.")