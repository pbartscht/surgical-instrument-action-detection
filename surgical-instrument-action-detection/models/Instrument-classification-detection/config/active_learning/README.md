# Active Learning Pipeline for Surgical Instrument Detection

This module implements an active learning pipeline for improving surgical instrument detection in laparoscopic videos using YOLO, specifically designed for the CholecT50 dataset.

## Prerequisites

- Python 3.8+
- Ultralytics YOLO
- OpenCV
- PyYAML
- CVAT.ai account

## Configuration

The active learning pipeline is configured through `instrument_config.yaml`

## Active Learning Workflow

### 1. Generate Predictions

Run the model predictor script:

```bash
python src/active_learning/model_predictor.py --config config/active_learning/instrument_config.yaml
```

### 2. Annotation Refinement in CVAT

1. Create new task in CVAT.ai
2. Upload target video frames (e.g., VID26 from CholeCT50 Dataset)
3. Configure instrument labels
4. Import generated predictions ("cvat_annotations.zip") from pretrained model via "Upload annotations" (in CVAT select annotation format YOLO 1.1)
5. Refine predictions within CVAT to get faster the correct labels
6. Export corrected annotations

### 3. Model Retraining

Retrain the model using the refined annotations to improve detection performance for surgical instruments in laparscopic cholecystectomies. 
