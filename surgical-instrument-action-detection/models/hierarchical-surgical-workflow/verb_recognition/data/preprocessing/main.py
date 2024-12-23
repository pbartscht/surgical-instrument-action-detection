from ultralytics import YOLO
import os
from core.core_processing import process_all_videos

if __name__ == "__main__":
    dataset_dir = "/data/Bartscht"
    verbs_dir = os.path.join(dataset_dir, "TestVerbs")
    
    model = YOLO('/home/Bartscht/YOLO/runs/detect/train35/weights/best.pt')
    process_all_videos(model, dataset_dir, verbs_dir)