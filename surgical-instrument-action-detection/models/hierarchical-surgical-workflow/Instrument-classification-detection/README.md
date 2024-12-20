# Surgical Instrument Classification and Detection

## Introduction
A computer vision pipeline for detecting and classifying surgical instruments in laparoscopic surgery videos is implemented in this project. Continuous improvement in detection accuracy is achieved by combining traditional machine learning approaches with active learning techniques. The system has been particularly designed for processing surgical video data, making it valuable for applications in computer-assisted surgery and surgical workflow analysis.

The pipeline follows a two-stage approach:
1. Initial training using a foundational dataset to establish baseline detection capabilities
2. Continuous improvement through active learning, where the model's predictions are refined through expert annotation

## Workflow

### 1. Data Preparation
- Follow the instructions in [`data/README.md`](data/README.md) for:
  - Downloading required datasets (M2CAI16-Tool-Locations and CholecT50)
  - Proper dataset organization 
  - Data conversion procedures for M2CAI16-Tool-Locations look under [`data/preprocessing/README.md`](data/preprocessing/README.md)

### 2. Model Training
```bash
python src/train_yolo.py
```
This script is used for:
1. Initial model training with the M2CAI16-Tool-Locations dataset
2. Subsequent retraining during active learning iterations with refined annotations
Important: Complete initial instrument model training before proceeding to verb recognition.

### 3. Active Learning Pipeline
Refer to [`config/active_learning/README.md`](config/active_learning/README.md) for:

1. Model prediction generation for unknown CholeCT50 images
2. Annotation refinement using CVAT.ai
3. Model retraining process using the same training script with updated CholeCT50 datasets positions.

### Prerequisites

- Python 3.8+
- Ultralytics YOLO
- OpenCV
- PyYAML
- CVAT.ai account

For detailed setup instructions and complete documentation of each component, please refer to the respective README files in each directory.
Citations
This work builds upon:
- [1] Jin, Amy, et al. "Tool detection and operative skill assessment in surgical videos using region-based convolutional neural networks." 2018 IEEE winter conference on applications of computer vision (WACV). IEEE, 2018.

- [2] C.I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux, N. Padoy. Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. Medical Image Analysis, 78 (2022) 102433.
Note: Access to the dataset is granted after completing the request form. 